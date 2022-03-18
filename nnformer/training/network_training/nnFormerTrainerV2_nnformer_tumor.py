#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from typing import Tuple


import numpy as np
import torch
from nnformer.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnformer.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnformer.utilities.to_torch import maybe_to_torch, to_cuda
from nnformer.network_architecture.nnFormer_tumor import nnFormer
from nnformer.network_architecture.initialization import InitWeights_He
from nnformer.network_architecture.neural_network import SegmentationNetwork
from nnformer.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnformer.training.dataloading.dataset_loading import unpack_dataset
from nnformer.training.network_training.nnFormerTrainer import nnFormerTrainer
from nnformer.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnformer.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *


    
class nnFormerTrainerV2_nnformer_tumor(nnFormerTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True
        self.load_pretrain_weight=True
        
        self.load_plans_file()    
        
        if len(self.plans['plans_per_stage'])==2:
            Stage=1
        else:
            Stage=0
            
        self.crop_size=self.plans['plans_per_stage'][Stage]['patch_size']
        self.input_channels=self.plans['num_modalities']
        self.num_classes=self.plans['num_classes'] + 1
        self.conv_op=nn.Conv3d
        
        self.embedding_dim=96
        self.depths=[2, 2, 2, 2]
        self.num_heads=[3, 6, 12, 24]
        self.embedding_patch_size=[4,4,4]
        self.window_size=[4,4,8,4]
        
        self.deep_supervision=False
    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()
            
            self.process_plans(self.plans)

            self.setup_DA_params()
            if self.deep_supervision:
                ################# Here we wrap the loss for deep supervision ############
                # we need to know the number of outputs of the network
                net_numpool = len(self.net_num_pool_op_kernel_sizes)

                # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                # this gives higher resolution outputs more weight in the loss
                weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                #mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
                #weights[~mask] = 0
                weights = weights / weights.sum()
                print(weights)
                self.ds_loss_weights = weights
                # now wrap the loss
                self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
                ################# END ###################
            
            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +"_stage%d" % self.stage)
            seeds_train = np.random.random_integers(0, 99999, self.data_aug_params.get('num_threads'))
            seeds_val = np.random.random_integers(0, 99999, max(self.data_aug_params.get('num_threads') // 2, 1))                         
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales if self.deep_supervision else None,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                    seeds_train=seeds_train,
                    seeds_val=seeds_val
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
  
      
        
        self.network=nnFormer(crop_size=self.crop_size,
                                embedding_dim=self.embedding_dim,
                                input_channels=self.input_channels,
                                num_classes=self.num_classes,
                                conv_op=self.conv_op,
                                depths=self.depths,
                                num_heads=self.num_heads,
                                patch_size=self.embedding_patch_size,
                                window_size=self.window_size,
                                deep_supervision=self.deep_supervision)
        if self.load_pretrain_weight:
            checkpoint = torch.load("/home/xychen/jsguo/weight/tumor_pretrain.model", map_location='cpu')
            ck={}
            
            for i in self.network.state_dict():
                if i in checkpoint:
                    print(i)
                    ck.update({i:checkpoint[i]})
                else:
                    ck.update({i:self.network.state_dict()[i]})
            self.network.load_state_dict(ck)
            print('I am using the pre_train weight!!')
        
     
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        if self.deep_supervision:
            target = target[0]
            output = output[0]
        else:
            target = target
            output = output
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            splits[self.fold]['train']=np.array(['BRATS_001', 'BRATS_002', 'BRATS_003', 'BRATS_004', 'BRATS_005',
       'BRATS_006', 'BRATS_007', 'BRATS_008', 'BRATS_009', 'BRATS_010',
       'BRATS_013', 'BRATS_014', 'BRATS_015', 'BRATS_016', 'BRATS_017',
       'BRATS_019', 'BRATS_022', 'BRATS_023', 'BRATS_024', 'BRATS_025',
       'BRATS_026', 'BRATS_027', 'BRATS_030', 'BRATS_031', 'BRATS_033',
       'BRATS_035', 'BRATS_037', 'BRATS_038', 'BRATS_039', 'BRATS_040',
       'BRATS_042', 'BRATS_043', 'BRATS_044', 'BRATS_045', 'BRATS_046',
       'BRATS_048', 'BRATS_050', 'BRATS_051', 'BRATS_052', 'BRATS_054',
       'BRATS_055', 'BRATS_060', 'BRATS_061', 'BRATS_062', 'BRATS_063',
       'BRATS_064', 'BRATS_065', 'BRATS_066', 'BRATS_067', 'BRATS_068',
       'BRATS_070', 'BRATS_072', 'BRATS_073', 'BRATS_074', 'BRATS_075',
       'BRATS_078', 'BRATS_079', 'BRATS_080', 'BRATS_081', 'BRATS_082',
       'BRATS_083', 'BRATS_084', 'BRATS_085', 'BRATS_086', 'BRATS_087',
       'BRATS_088', 'BRATS_091', 'BRATS_093', 'BRATS_094', 'BRATS_096',
       'BRATS_097', 'BRATS_098', 'BRATS_100', 'BRATS_101', 'BRATS_102',
       'BRATS_104', 'BRATS_108', 'BRATS_110', 'BRATS_111', 'BRATS_112',
       'BRATS_115', 'BRATS_116', 'BRATS_117', 'BRATS_119', 'BRATS_120',
       'BRATS_121', 'BRATS_122', 'BRATS_123', 'BRATS_125', 'BRATS_126',
       'BRATS_127', 'BRATS_128', 'BRATS_129', 'BRATS_130', 'BRATS_131',
       'BRATS_132', 'BRATS_133', 'BRATS_134', 'BRATS_135', 'BRATS_136',
       'BRATS_137', 'BRATS_138', 'BRATS_140', 'BRATS_141', 'BRATS_142',
       'BRATS_143', 'BRATS_144', 'BRATS_146', 'BRATS_148', 'BRATS_149',
       'BRATS_150', 'BRATS_153', 'BRATS_154', 'BRATS_155', 'BRATS_158',
       'BRATS_159', 'BRATS_160', 'BRATS_162', 'BRATS_163', 'BRATS_164',
       'BRATS_165', 'BRATS_166', 'BRATS_167', 'BRATS_168', 'BRATS_169',
       'BRATS_170', 'BRATS_171', 'BRATS_173', 'BRATS_174', 'BRATS_175',
       'BRATS_177', 'BRATS_178', 'BRATS_179', 'BRATS_180', 'BRATS_182',
       'BRATS_183', 'BRATS_184', 'BRATS_185', 'BRATS_186', 'BRATS_187',
       'BRATS_188', 'BRATS_189', 'BRATS_191', 'BRATS_192', 'BRATS_193',
       'BRATS_195', 'BRATS_197', 'BRATS_199', 'BRATS_200', 'BRATS_201',
       'BRATS_202', 'BRATS_203', 'BRATS_206', 'BRATS_207', 'BRATS_208',
       'BRATS_210', 'BRATS_211', 'BRATS_212', 'BRATS_213', 'BRATS_214',
       'BRATS_215', 'BRATS_216', 'BRATS_217', 'BRATS_218', 'BRATS_219',
       'BRATS_222', 'BRATS_223', 'BRATS_224', 'BRATS_225', 'BRATS_226',
       'BRATS_228', 'BRATS_229', 'BRATS_230', 'BRATS_231', 'BRATS_232',
       'BRATS_233', 'BRATS_236', 'BRATS_237', 'BRATS_238', 'BRATS_239',
       'BRATS_241', 'BRATS_243', 'BRATS_244', 'BRATS_246', 'BRATS_247',
       'BRATS_248', 'BRATS_249', 'BRATS_251', 'BRATS_252', 'BRATS_253',
       'BRATS_254', 'BRATS_255', 'BRATS_258', 'BRATS_259', 'BRATS_261',
       'BRATS_262', 'BRATS_263', 'BRATS_264', 'BRATS_265', 'BRATS_266',
       'BRATS_267', 'BRATS_268', 'BRATS_272', 'BRATS_273', 'BRATS_274',
       'BRATS_275', 'BRATS_276', 'BRATS_277', 'BRATS_278', 'BRATS_279',
       'BRATS_280', 'BRATS_283', 'BRATS_284', 'BRATS_285', 'BRATS_286',
       'BRATS_288', 'BRATS_290', 'BRATS_293', 'BRATS_294', 'BRATS_296',
       'BRATS_297', 'BRATS_298', 'BRATS_299', 'BRATS_300', 'BRATS_301',
       'BRATS_302', 'BRATS_303', 'BRATS_304', 'BRATS_306', 'BRATS_307',
       'BRATS_308', 'BRATS_309', 'BRATS_311', 'BRATS_312', 'BRATS_313',
       'BRATS_315', 'BRATS_316', 'BRATS_317', 'BRATS_318', 'BRATS_319',
       'BRATS_320', 'BRATS_321', 'BRATS_322', 'BRATS_324', 'BRATS_326',
       'BRATS_328', 'BRATS_329', 'BRATS_332', 'BRATS_334', 'BRATS_335',
       'BRATS_336', 'BRATS_338', 'BRATS_339', 'BRATS_340', 'BRATS_341',
       'BRATS_342', 'BRATS_343', 'BRATS_344', 'BRATS_345', 'BRATS_347',
       'BRATS_348', 'BRATS_349', 'BRATS_351', 'BRATS_353', 'BRATS_354',
       'BRATS_355', 'BRATS_356', 'BRATS_357', 'BRATS_358', 'BRATS_359',
       'BRATS_360', 'BRATS_363', 'BRATS_364', 'BRATS_365', 'BRATS_366',
       'BRATS_367', 'BRATS_368', 'BRATS_369', 'BRATS_370', 'BRATS_371',
       'BRATS_372', 'BRATS_373', 'BRATS_374', 'BRATS_375', 'BRATS_376',
       'BRATS_377', 'BRATS_378', 'BRATS_379', 'BRATS_380', 'BRATS_381',
       'BRATS_383', 'BRATS_384', 'BRATS_385', 'BRATS_386', 'BRATS_387',
       'BRATS_388', 'BRATS_390', 'BRATS_391', 'BRATS_392', 'BRATS_393',
       'BRATS_394', 'BRATS_395', 'BRATS_396', 'BRATS_398', 'BRATS_399',
       'BRATS_401', 'BRATS_403', 'BRATS_404', 'BRATS_405', 'BRATS_407',
       'BRATS_408', 'BRATS_409', 'BRATS_410', 'BRATS_411', 'BRATS_412',
       'BRATS_413', 'BRATS_414', 'BRATS_415', 'BRATS_417', 'BRATS_418',
       'BRATS_419', 'BRATS_420', 'BRATS_421', 'BRATS_422', 'BRATS_423',
       'BRATS_424', 'BRATS_426', 'BRATS_428', 'BRATS_429', 'BRATS_430',
       'BRATS_431', 'BRATS_433', 'BRATS_434', 'BRATS_435', 'BRATS_436',
       'BRATS_437', 'BRATS_438', 'BRATS_439', 'BRATS_441', 'BRATS_442',
       'BRATS_443', 'BRATS_444', 'BRATS_445', 'BRATS_446', 'BRATS_449',
       'BRATS_451', 'BRATS_452', 'BRATS_453', 'BRATS_454', 'BRATS_455',
       'BRATS_457', 'BRATS_458', 'BRATS_459', 'BRATS_460', 'BRATS_463',
       'BRATS_464', 'BRATS_466', 'BRATS_467', 'BRATS_468', 'BRATS_469',
       'BRATS_470', 'BRATS_472', 'BRATS_475', 'BRATS_477', 'BRATS_478',
       'BRATS_481', 'BRATS_482', 'BRATS_483','BRATS_400', 'BRATS_402',
       'BRATS_406', 'BRATS_416', 'BRATS_427', 'BRATS_440', 'BRATS_447',
       'BRATS_448', 'BRATS_456', 'BRATS_461', 'BRATS_462', 'BRATS_465',
       'BRATS_471', 'BRATS_473', 'BRATS_474', 'BRATS_476', 'BRATS_479',
       'BRATS_480', 'BRATS_484'])
            splits[self.fold]['val']=np.array(['BRATS_011', 'BRATS_012', 'BRATS_018', 'BRATS_020', 'BRATS_021',
       'BRATS_028', 'BRATS_029', 'BRATS_032', 'BRATS_034', 'BRATS_036',
       'BRATS_041', 'BRATS_047', 'BRATS_049', 'BRATS_053', 'BRATS_056',
       'BRATS_057', 'BRATS_069', 'BRATS_071', 'BRATS_089', 'BRATS_090',
       'BRATS_092', 'BRATS_095', 'BRATS_103', 'BRATS_105', 'BRATS_106',
       'BRATS_107', 'BRATS_109', 'BRATS_118', 'BRATS_145', 'BRATS_147',
       'BRATS_156', 'BRATS_161', 'BRATS_172', 'BRATS_176', 'BRATS_181',
       'BRATS_194', 'BRATS_196', 'BRATS_198', 'BRATS_204', 'BRATS_205',
       'BRATS_209', 'BRATS_220', 'BRATS_221', 'BRATS_227', 'BRATS_234',
       'BRATS_235', 'BRATS_245', 'BRATS_250', 'BRATS_256', 'BRATS_257',
       'BRATS_260', 'BRATS_269', 'BRATS_270', 'BRATS_271', 'BRATS_281',
       'BRATS_282', 'BRATS_287', 'BRATS_289', 'BRATS_291', 'BRATS_292',
       'BRATS_310', 'BRATS_314', 'BRATS_323', 'BRATS_327', 'BRATS_330',
       'BRATS_333', 'BRATS_337', 'BRATS_346', 'BRATS_350', 'BRATS_352',
       'BRATS_361', 'BRATS_382', 'BRATS_397'])
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        if self.deep_supervision:
            self.network.do_ds = True
        else:
            self.network.do_ds = False
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
