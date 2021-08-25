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


import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnformer.paths import network_training_output_dir

if __name__ == "__main__":
    # run collect_all_fold0_results_and_summarize_in_one_csv.py first
    summary_files_dir = join(network_training_output_dir, "summary_jsons_fold0_new")
    output_file = join(network_training_output_dir, "summary.csv")

    folds = (0, )
    folds_str = ""
    for f in folds:
        folds_str += str(f)

    plans = "nnFormerPlans"

    overwrite_plans = {
        'nnFormerTrainerV2_2': ["nnFormerPlans", "nnFormerPlansisoPatchesInVoxels"], # r
        'nnFormerTrainerV2': ["nnFormerPlansnonCT", "nnFormerPlansCT2", "nnFormerPlansallConv3x3",
                            "nnFormerPlansfixedisoPatchesInVoxels", "nnFormerPlanstargetSpacingForAnisoAxis",
                            "nnFormerPlanspoolBasedOnSpacing", "nnFormerPlansfixedisoPatchesInmm", "nnFormerPlansv2.1"],
        'nnFormerTrainerV2_warmup': ["nnFormerPlans", "nnFormerPlansv2.1", "nnFormerPlansv2.1_big", "nnFormerPlansv2.1_verybig"],
        'nnFormerTrainerV2_cycleAtEnd': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_cycleAtEnd2': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_reduceMomentumDuringTraining': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_graduallyTransitionFromCEToDice': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_independentScalePerAxis': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_Mish': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_Ranger_lr3en4': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_fp32': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_GN': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_momentum098': ["nnFormerPlans", "nnFormerPlansv2.1"],
        'nnFormerTrainerV2_momentum09': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_DP': ["nnFormerPlansv2.1_verybig"],
        'nnFormerTrainerV2_DDP': ["nnFormerPlansv2.1_verybig"],
        'nnFormerTrainerV2_FRN': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_resample33': ["nnFormerPlansv2.3"],
        'nnFormerTrainerV2_O2': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_ResencUNet': ["nnFormerPlans_FabiansResUNet_v2.1"],
        'nnFormerTrainerV2_DA2': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_allConv3x3': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_ForceBD': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_ForceSD': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_LReLU_slope_2en1': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_lReLU_convReLUIN': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_ReLU': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_ReLU_biasInSegOutput': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_ReLU_convReLUIN': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_lReLU_biasInSegOutput': ["nnFormerPlansv2.1"],
        #'nnFormerTrainerV2_Loss_MCC': ["nnFormerPlansv2.1"],
        #'nnFormerTrainerV2_Loss_MCCnoBG': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_Loss_DicewithBG': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_Loss_Dice_LR1en3': ["nnFormerPlansv2.1"],
        'nnFormerTrainerV2_Loss_Dice': ["nnFormerPlans", "nnFormerPlansv2.1"],
        'nnFormerTrainerV2_Loss_DicewithBG_LR1en3': ["nnFormerPlansv2.1"],
        # 'nnFormerTrainerV2_fp32': ["nnFormerPlansv2.1"],
        # 'nnFormerTrainerV2_fp32': ["nnFormerPlansv2.1"],
        # 'nnFormerTrainerV2_fp32': ["nnFormerPlansv2.1"],
        # 'nnFormerTrainerV2_fp32': ["nnFormerPlansv2.1"],
        # 'nnFormerTrainerV2_fp32': ["nnFormerPlansv2.1"],

    }

    trainers = ['nnFormerTrainer'] + ['nnFormerTrainerNewCandidate%d' % i for i in range(1, 28)] + [
        'nnFormerTrainerNewCandidate24_2',
        'nnFormerTrainerNewCandidate24_3',
        'nnFormerTrainerNewCandidate26_2',
        'nnFormerTrainerNewCandidate27_2',
        'nnFormerTrainerNewCandidate23_always3DDA',
        'nnFormerTrainerNewCandidate23_corrInit',
        'nnFormerTrainerNewCandidate23_noOversampling',
        'nnFormerTrainerNewCandidate23_softDS',
        'nnFormerTrainerNewCandidate23_softDS2',
        'nnFormerTrainerNewCandidate23_softDS3',
        'nnFormerTrainerNewCandidate23_softDS4',
        'nnFormerTrainerNewCandidate23_2_fp16',
        'nnFormerTrainerNewCandidate23_2',
        'nnFormerTrainerVer2',
        'nnFormerTrainerV2_2',
        'nnFormerTrainerV2_3',
        'nnFormerTrainerV2_3_CE_GDL',
        'nnFormerTrainerV2_3_dcTopk10',
        'nnFormerTrainerV2_3_dcTopk20',
        'nnFormerTrainerV2_3_fp16',
        'nnFormerTrainerV2_3_softDS4',
        'nnFormerTrainerV2_3_softDS4_clean',
        'nnFormerTrainerV2_3_softDS4_clean_improvedDA',
        'nnFormerTrainerV2_3_softDS4_clean_improvedDA_newElDef',
        'nnFormerTrainerV2_3_softDS4_radam',
        'nnFormerTrainerV2_3_softDS4_radam_lowerLR',

        'nnFormerTrainerV2_2_schedule',
        'nnFormerTrainerV2_2_schedule2',
        'nnFormerTrainerV2_2_clean',
        'nnFormerTrainerV2_2_clean_improvedDA_newElDef',

        'nnFormerTrainerV2_2_fixes', # running
        'nnFormerTrainerV2_BN', # running
        'nnFormerTrainerV2_noDeepSupervision', # running
        'nnFormerTrainerV2_softDeepSupervision', # running
        'nnFormerTrainerV2_noDataAugmentation', # running
        'nnFormerTrainerV2_Loss_CE', # running
        'nnFormerTrainerV2_Loss_CEGDL',
        'nnFormerTrainerV2_Loss_Dice',
        'nnFormerTrainerV2_Loss_DiceTopK10',
        'nnFormerTrainerV2_Loss_TopK10',
        'nnFormerTrainerV2_Adam', # running
        'nnFormerTrainerV2_Adam_nnFormerTrainerlr', # running
        'nnFormerTrainerV2_SGD_ReduceOnPlateau', # running
        'nnFormerTrainerV2_SGD_lr1en1', # running
        'nnFormerTrainerV2_SGD_lr1en3', # running
        'nnFormerTrainerV2_fixedNonlin', # running
        'nnFormerTrainerV2_GeLU', # running
        'nnFormerTrainerV2_3ConvPerStage',
        'nnFormerTrainerV2_NoNormalization',
        'nnFormerTrainerV2_Adam_ReduceOnPlateau',
        'nnFormerTrainerV2_fp16',
        'nnFormerTrainerV2', # see overwrite_plans
        'nnFormerTrainerV2_noMirroring',
        'nnFormerTrainerV2_momentum09',
        'nnFormerTrainerV2_momentum095',
        'nnFormerTrainerV2_momentum098',
        'nnFormerTrainerV2_warmup',
        'nnFormerTrainerV2_Loss_Dice_LR1en3',
        'nnFormerTrainerV2_NoNormalization_lr1en3',
        'nnFormerTrainerV2_Loss_Dice_squared',
        'nnFormerTrainerV2_newElDef',
        'nnFormerTrainerV2_fp32',
        'nnFormerTrainerV2_cycleAtEnd',
        'nnFormerTrainerV2_reduceMomentumDuringTraining',
        'nnFormerTrainerV2_graduallyTransitionFromCEToDice',
        'nnFormerTrainerV2_insaneDA',
        'nnFormerTrainerV2_independentScalePerAxis',
        'nnFormerTrainerV2_Mish',
        'nnFormerTrainerV2_Ranger_lr3en4',
        'nnFormerTrainerV2_cycleAtEnd2',
        'nnFormerTrainerV2_GN',
        'nnFormerTrainerV2_DP',
        'nnFormerTrainerV2_FRN',
        'nnFormerTrainerV2_resample33',
        'nnFormerTrainerV2_O2',
        'nnFormerTrainerV2_ResencUNet',
        'nnFormerTrainerV2_DA2',
        'nnFormerTrainerV2_allConv3x3',
        'nnFormerTrainerV2_ForceBD',
        'nnFormerTrainerV2_ForceSD',
        'nnFormerTrainerV2_ReLU',
        'nnFormerTrainerV2_LReLU_slope_2en1',
        'nnFormerTrainerV2_lReLU_convReLUIN',
        'nnFormerTrainerV2_ReLU_biasInSegOutput',
        'nnFormerTrainerV2_ReLU_convReLUIN',
        'nnFormerTrainerV2_lReLU_biasInSegOutput',
        'nnFormerTrainerV2_Loss_DicewithBG_LR1en3',
        #'nnFormerTrainerV2_Loss_MCCnoBG',
        'nnFormerTrainerV2_Loss_DicewithBG',
        # 'nnFormerTrainerV2_Loss_Dice_LR1en3',
        # 'nnFormerTrainerV2_Ranger_lr3en4',
        # 'nnFormerTrainerV2_Ranger_lr3en4',
        # 'nnFormerTrainerV2_Ranger_lr3en4',
        # 'nnFormerTrainerV2_Ranger_lr3en4',
        # 'nnFormerTrainerV2_Ranger_lr3en4',
        # 'nnFormerTrainerV2_Ranger_lr3en4',
        # 'nnFormerTrainerV2_Ranger_lr3en4',
        # 'nnFormerTrainerV2_Ranger_lr3en4',
        # 'nnFormerTrainerV2_Ranger_lr3en4',
        # 'nnFormerTrainerV2_Ranger_lr3en4',
        # 'nnFormerTrainerV2_Ranger_lr3en4',
        # 'nnFormerTrainerV2_Ranger_lr3en4',
        # 'nnFormerTrainerV2_Ranger_lr3en4',
    ]

    datasets = \
        {"Task001_BrainTumour": ("3d_fullres", ),
        "Task002_Heart": ("3d_fullres",),
        #"Task024_Promise": ("3d_fullres",),
        #"Task027_ACDC": ("3d_fullres",),
        "Task003_Liver": ("3d_fullres", "3d_lowres"),
        "Task004_Hippocampus": ("3d_fullres",),
        "Task005_Prostate": ("3d_fullres",),
        "Task006_Lung": ("3d_fullres", "3d_lowres"),
        "Task007_Pancreas": ("3d_fullres", "3d_lowres"),
        "Task008_HepaticVessel": ("3d_fullres", "3d_lowres"),
        "Task009_Spleen": ("3d_fullres", "3d_lowres"),
        "Task010_Colon": ("3d_fullres", "3d_lowres"),}

    expected_validation_folder = "validation_raw"
    alternative_validation_folder = "validation"
    alternative_alternative_validation_folder = "validation_tiledTrue_doMirror_True"

    interested_in = "mean"

    result_per_dataset = {}
    for d in datasets:
        result_per_dataset[d] = {}
        for c in datasets[d]:
            result_per_dataset[d][c] = []

    valid_trainers = []
    all_trainers = []

    with open(output_file, 'w') as f:
        f.write("trainer,")
        for t in datasets.keys():
            s = t[4:7]
            for c in datasets[t]:
                s1 = s + "_" + c[3]
                f.write("%s," % s1)
        f.write("\n")

        for trainer in trainers:
            trainer_plans = [plans]
            if trainer in overwrite_plans.keys():
                trainer_plans = overwrite_plans[trainer]

            result_per_dataset_here = {}
            for d in datasets:
                result_per_dataset_here[d] = {}

            for p in trainer_plans:
                name = "%s__%s" % (trainer, p)
                all_present = True
                all_trainers.append(name)

                f.write("%s," % name)
                for dataset in datasets.keys():
                    for configuration in datasets[dataset]:
                        summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, expected_validation_folder, folds_str))
                        if not isfile(summary_file):
                            summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, alternative_validation_folder, folds_str))
                            if not isfile(summary_file):
                                summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (
                                dataset, configuration, trainer, p, alternative_alternative_validation_folder, folds_str))
                                if not isfile(summary_file):
                                    all_present = False
                                    print(name, dataset, configuration, "has missing summary file")
                        if isfile(summary_file):
                            result = load_json(summary_file)['results'][interested_in]['mean']['Dice']
                            result_per_dataset_here[dataset][configuration] = result
                            f.write("%02.4f," % result)
                        else:
                            f.write("NA,")
                            result_per_dataset_here[dataset][configuration] = 0

                f.write("\n")

                if True:
                    valid_trainers.append(name)
                    for d in datasets:
                        for c in datasets[d]:
                            result_per_dataset[d][c].append(result_per_dataset_here[d][c])

    invalid_trainers = [i for i in all_trainers if i not in valid_trainers]

    num_valid = len(valid_trainers)
    num_datasets = len(datasets.keys())
    # create an array that is trainer x dataset. If more than one configuration is there then use the best metric across the two
    all_res = np.zeros((num_valid, num_datasets))
    for j, d in enumerate(datasets.keys()):
        ks = list(result_per_dataset[d].keys())
        tmp = result_per_dataset[d][ks[0]]
        for k in ks[1:]:
            for i in range(len(tmp)):
                tmp[i] = max(tmp[i], result_per_dataset[d][k][i])
        all_res[:, j] = tmp

    ranks_arr = np.zeros_like(all_res)
    for d in range(ranks_arr.shape[1]):
        temp = np.argsort(all_res[:, d])[::-1] # inverse because we want the highest dice to be rank0
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))

        ranks_arr[:, d] = ranks

    mn = np.mean(ranks_arr, 1)
    for i in np.argsort(mn):
        print(mn[i], valid_trainers[i])

    print()
    print(valid_trainers[np.argmin(mn)])
