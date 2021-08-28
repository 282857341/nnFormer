from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="task id")
    args = parser.parse_args()
    if args.task=='2':
        input_file = '../DATASET/nnFormer_preprocessed/Task002_Synapse/nnFormerPlansv2.1_plans_3D.pkl'
        output_file = '../DATASET/nnFormer_preprocessed/Task002_Synapse/nnFormerPlansv2.1_Synapse_plans_3D.pkl'
        a = load_pickle(input_file)
        
        a['plans_per_stage'][1]['patch_size']=np.array([64,128,128])
        a['plans_per_stage'][1]['pool_op_kernel_sizes']=[[2,2,2],[2,2,2],[2,2,2],[2,2,2]]
        a['plans_per_stage'][1]['conv_kernel_sizes']=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]]
        save_pickle(a, output_file)
        
    elif args.task=='1':
        input_file = './DATASET/nnFormer_preprocessed/Task001_ACDC/nnFormerPlansv2.1_plans_3D.pkl'
        output_file = './DATASET/nnFormer_preprocessed/Task001_ACDC/nnFormerPlansv2.1_ACDC_plans_3D.pkl'
        a = load_pickle(input_file)
        
        a['plans_per_stage'][0]['patch_size']=np.array([14,160,160])
        a['plans_per_stage'][0]['pool_op_kernel_sizes']=[[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]]
        a['plans_per_stage'][0]['conv_kernel_sizes']=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]]
        save_pickle(a, output_file)
    print(output_file)
