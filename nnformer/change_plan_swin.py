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
        
        # Here is to change the dataset division
        split_file=input_file.replace('nnFormerPlansv2.1_plans_3D','splits_final')
        b = load_pickle(split_file)
        b[0]['train']=np.array(['img0006','img0007' ,'img0009', 'img0010', 'img0021' ,'img0023' ,'img0024','img0026' ,'img0027' ,'img0031', 'img0033' ,'img0034' \
                                ,'img0039', 'img0040','img0005', 'img0028', 'img0030', 'img0037'])
        b[0]['val']=np.array(['img0001', 'img0002', 'img0003', 'img0004', 'img0008', 'img0022','img0025', 'img0029', 'img0032', 'img0035', 'img0036', 'img0038'])
        save_pickle(b,split_file)
        
    elif args.task=='1':
        input_file = './DATASET/nnFormer_preprocessed/Task001_ACDC/nnFormerPlansv2.1_plans_3D.pkl'
        output_file = './DATASET/nnFormer_preprocessed/Task001_ACDC/nnFormerPlansv2.1_ACDC_plans_3D.pkl'
        a = load_pickle(input_file)
        
        a['plans_per_stage'][0]['patch_size']=np.array([14,160,160])
        a['plans_per_stage'][0]['pool_op_kernel_sizes']=[[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]]
        a['plans_per_stage'][0]['conv_kernel_sizes']=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]]
        
        # The train list and val list of the acdc dataset is long, so I put it in the list_acdc.txt, you can change the dataset division as the Synapse task
        save_pickle(a, output_file)
    print(output_file)
