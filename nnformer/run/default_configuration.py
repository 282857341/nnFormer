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


import nnformer
from nnformer.paths import network_training_output_dir, preprocessing_output_dir, default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnformer.experiment_planning.summarize_plans import summarize_plans
from nnformer.training.model_restore import recursive_find_python_class
import numpy as np
import pickle

def get_configuration_from_output_folder(folder):
    # split off network_training_output_dir
    folder = folder[len(network_training_output_dir):]
    if folder.startswith("/"):
        folder = folder[1:]

    configuration, task, trainer_and_plans_identifier = folder.split("/")
    trainer, plans_identifier = trainer_and_plans_identifier.split("__")
    return configuration, task, trainer, plans_identifier


def get_default_configuration(network, task, network_trainer, plans_identifier=default_plans_identifier,
                              search_in=(nnformer.__path__[0], "training", "network_training"),
                              base_module='nnformer.training.network_training'):
    assert network in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'], \
        "network can only be one of the following: \'3d\', \'3d_lowres\', \'3d_fullres\', \'3d_cascade_fullres\'"

    dataset_directory = join(preprocessing_output_dir, task)

    if network == '2d':
        plans_file = join(preprocessing_output_dir, task, plans_identifier + "_plans_2D.pkl")
    else:
        plans_file = join(preprocessing_output_dir, task, plans_identifier + "_plans_3D.pkl")
             
    plans = load_pickle(plans_file)
    # Maybe have two kinds of plans,choose the later one 
    if len(plans['plans_per_stage'])==2:
        Stage=1
    else:
        Stage=0
    if task=='Task001_ACDC':
        plans['plans_per_stage'][Stage]['batch_size']=4
        plans['plans_per_stage'][Stage]['patch_size']=np.array([14,160,160])
        pickle_file = open(plans_file,'wb')
        pickle.dump(plans, pickle_file)
        pickle_file.close()

    elif task=='Task002_Synapse':
        plans['plans_per_stage'][Stage]['batch_size']=2
        plans['plans_per_stage'][Stage]['patch_size']=np.array([64,128,128])
        plans['plans_per_stage'][Stage]['pool_op_kernel_sizes']=[[2,2,2],[2,2,2],[2,2,2]] # for deep supervision
        pickle_file = open(plans_file,'wb')
        pickle.dump(plans, pickle_file)
        pickle_file.close()
    elif task=='Task003_tumor':
        plans['plans_per_stage'][Stage]['batch_size']=2
        plans['plans_per_stage'][Stage]['patch_size']=np.array([128,128,128])
        pickle_file = open(plans_file,'wb')
        pickle.dump(plans, pickle_file)
        pickle_file.close()
    possible_stages = list(plans['plans_per_stage'].keys())

    if (network == '3d_cascade_fullres' or network == "3d_lowres") and len(possible_stages) == 1:
        raise RuntimeError("3d_lowres/3d_cascade_fullres only applies if there is more than one stage. This task does "
                           "not require the cascade. Run 3d_fullres instead")


    if network == '2d' or network == "3d_lowres":
        stage = 0
    else:
        stage = possible_stages[-1]

    trainer_class = recursive_find_python_class([join(*search_in)], network_trainer,
                                                current_module=base_module)

    output_folder_name = join(network_training_output_dir, network, task, network_trainer + "__" + plans_identifier)

    print("###############################################")
    print("I am running the following nnFormer: %s" % network)
    print("My trainer class is: ", trainer_class)
    print("For that I will be using the following configuration:")
    summarize_plans(plans_file)
    print("I am using stage %d from these plans" % stage)

    if (network == '2d' or len(possible_stages) > 1) and not network == '3d_lowres':
        batch_dice = True
        print("I am using batch dice + CE loss")
    else:
        batch_dice = False
        print("I am using sample dice + CE loss")

    print("\nI am using data from this folder: ", join(dataset_directory, plans['data_identifier']))
    print("###############################################")
    return plans_file, output_folder_name, dataset_directory, batch_dice, stage, trainer_class
