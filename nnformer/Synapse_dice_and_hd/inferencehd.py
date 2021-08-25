import glob
import os
import SimpleITK as sitk
import numpy as np
from medpy import metric
#import surface_distance as surfdist

'''
def hausd95(mask_pred, mask_gt, spacing_mm_s=1):
    width_t, height_t, queue0_t, queue1_t = mask_gt.dataobj.shape
    width_p, height_p, queue0_p, queue1_p = mask_pred.dataobj.shape 
    
    mH = []
    if(queue1_t != queue1_p):
        return("Error,The two sets of data have different dimensions")
    else:
        for i in range(queue1_t):
            gt = mask_gt.dataobj[:,:,:,i]
            pred = mask_pred.dataobj[:,:,:,i]
            
            gt = gt.astype(np.bool)
            pred = pred.astype(np.bool)
    
            surface_distances = surfdist.compute_surface_distances(gt, pred, spacing_mm = spacing_mm_s)
            hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95) 
            mH.append(hd_dist_95)   
    return mH
'''

def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


    
def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())

def process_label(label):
    spleen = label == 1
    right_kidney = label == 2
    left_kidney = label == 3
    gallbladder = label == 4
    esophagus = label == 5
    liver = label == 6
    stomach = label == 7
    aorta = label == 8
    inferior_vena_cava = label == 9
    portal_vein_splenic_vein = label == 10
    pancreas = label == 11
    right_adrenal_gland = label == 12
    left_adrenal_gland = label == 13
   
    return spleen,right_kidney,left_kidney,gallbladder,esophagus,liver,stomach,aorta,inferior_vena_cava,portal_vein_splenic_vein,pancreas,right_adrenal_gland,left_adrenal_gland

def test(fold):
    '''
    def hd(lP,lT):
        if lP.sum() > 0 and lT.sum()>0:
            labelPred=sitk.GetImageFromArray(lP.astype(np.float32), isVector=False)
            labelTrue=sitk.GetImageFromArray(lT.astype(np.float32), isVector=False)
            print(labelTrue.GetSpacing())
            hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
            hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
            return hausdorffcomputer.GetHausdorffDistance()#
        else:
            return 0
    '''
    def hd(pred,gt):
        if pred.sum() > 0 and gt.sum()>0:
            hd95 = metric.binary.hd95(pred, gt)
            print(hd95)
            return  hd95
        else:
            return 0
    
    path='../DATASET/nnFormer_raw/nnFormer_raw_data/Task002_Synapse/'
    label_list=sorted(glob.glob(os.path.join(path,'labelsTs','*nii.gz')))
    infer_list=sorted(glob.glob(os.path.join(path,'inferTs',fold,'*nii.gz')))
    print("loading success...")
    print(label_list)
    print(infer_list)
    
    
    hd_spleen=[]
    hd_right_kidney=[]
    hd_left_kidney=[]
    hd_gallbladder=[]
    hd_esophagus=[]
    hd_liver=[]
    hd_stomach=[]
    hd_aorta=[]
    hd_inferior_vena_cava=[]
    hd_portal_vein_splenic_vein=[]
    hd_pancreas=[]
    hd_right_adrenal_gland=[]
    hd_left_adrenal_gland=[]
    
    file=path + 'inferTs/'+fold
    if not os.path.exists(file):
        os.makedirs(file)
    fw = open(file+'/avghd_test.txt', 'a')
    
    for label_path,infer_path in zip(label_list,infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label,infer = read_nii(label_path),read_nii(infer_path)
        label_spleen,label_right_kidney,label_left_kidney,label_gallbladder,label_esophagus,label_liver,label_stomach,label_aorta,label_inferior_vena_cava,label_portal_vein_splenic_vein,label_pancreas,label_right_adrenal_gland,label_left_adrenal_gland=process_label(label)
        
        
        infer_spleen,infer_right_kidney,infer_left_kidney,infer_gallbladder,infer_esophagus,infer_liver,infer_stomach,infer_aorta,infer_inferior_vena_cava,infer_portal_vein_splenic_vein,infer_pancreas,infer_right_adrenal_gland,infer_left_adrenal_gland=process_label(infer)

        
        hd_spleen.append(hd(infer_spleen,label_spleen))
        hd_right_kidney.append(hd(infer_right_kidney,label_right_kidney))
        hd_left_kidney.append(hd(infer_left_kidney,label_left_kidney))
        hd_gallbladder.append(hd(infer_gallbladder,label_gallbladder))
        hd_liver.append(hd(infer_liver,label_liver))
        hd_stomach.append(hd(infer_stomach,label_stomach))
        hd_aorta.append(hd(infer_aorta,label_aorta))
        hd_pancreas.append(hd(infer_pancreas,label_pancreas))

        fw.write('*'*20+'\n',)
        fw.write(infer_path.split('/')[-1]+'\n')
        fw.write('hd_spleen: {:.4f}\n'.format(hd_spleen[-1]))
        fw.write('hd_right_kidney: {:.4f}\n'.format(hd_right_kidney[-1]))
        fw.write('hd_left_kidney: {:.4f}\n'.format(hd_left_kidney[-1]))
        fw.write('hd_gallbladder: {:.4f}\n'.format(hd_gallbladder[-1]))
        fw.write('hd_liver: {:.4f}\n'.format(hd_liver[-1]))
        fw.write('hd_stomach: {:.4f}\n'.format(hd_stomach[-1]))
        fw.write('hd_aorta: {:.4f}\n'.format(hd_aorta[-1]))
        fw.write('hd_pancreas: {:.4f}\n'.format(hd_pancreas[-1]))
        fw.write('*'*20+'\n')
    
    
    
    fw.write('*'*20+'\n')
    fw.write('Mean_hd\n')
    fw.write('hd_spleen'+str(np.mean(hd_spleen))+'\n')
    fw.write('hd_right_kidney'+str(np.mean(hd_right_kidney))+'\n')
    fw.write('hd_left_kidney'+str(np.mean(hd_left_kidney))+'\n')
    fw.write('hd_gallbladder'+str(np.mean(hd_gallbladder))+'\n')
    fw.write('hd_liver'+str(np.mean(hd_liver))+'\n')
    fw.write('hd_stomach'+str(np.mean(hd_stomach))+'\n')
    fw.write('hd_aorta'+str(np.mean(hd_aorta))+'\n')
    fw.write('hd_pancreas'+str(np.mean(hd_pancreas))+'\n')
    fw.write('*'*20+'\n')
    
    hd=[]
    
    
    hd.append(np.mean(hd_spleen))
    hd.append(np.mean(hd_right_kidney))
    hd.append(np.mean(hd_left_kidney))
    hd.append(np.mean(hd_gallbladder))
    hd.append(np.mean(hd_liver))
    hd.append(np.mean(hd_stomach))
    hd.append(np.mean(hd_aorta))
    hd.append(np.mean(hd_pancreas))
    fw.write('hd:'+str(np.mean(hd))+'\n')
    print('done')

if __name__ == '__main__':
    fold='output'
    test(fold)
