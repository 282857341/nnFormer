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
    
    #path='/home/xychen/new_transformer/nnUNetFrame/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task002_Abdomen/nnUNetTrainerV2_swin_l_mix_head3__nnUNetPlansv2.1/fold_0/validation_raw/'
    path='/home/xychen/new_transformer/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task002_Abdomen/'
    label_list=sorted(glob.glob(os.path.join(path,'labelsTs','*nii.gz')))
    infer_list=sorted(glob.glob(os.path.join(path,'inferTs',fold,'*nii.gz')))
    #infer_list =sorted(glob.glob('/data3/jsguo/nnUNetFrame/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task004_Abdomen/nnUNetTrainerV2_swin_l_mix_win__nnUNetPlansv2.1/fold_0/validation_raw/*.nii.gz'))
    print("loading success...")
    print(label_list)
    print(infer_list)
    Dice_spleen=[]
    Dice_right_kidney=[]
    Dice_left_kidney=[]
    Dice_gallbladder=[]
    Dice_esophagus=[]
    Dice_liver=[]
    Dice_stomach=[]
    Dice_aorta=[]
    Dice_inferior_vena_cava=[]
    Dice_portal_vein_splenic_vein=[]
    Dice_pancreas=[]
    Dice_right_adrenal_gland=[]
    Dice_left_adrenal_gland=[]
    
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
    fw = open(file+'/avghd_test_dice.txt', 'a')
    
    for label_path,infer_path in zip(label_list,infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label,infer = read_nii(label_path),read_nii(infer_path)
        label_spleen,label_right_kidney,label_left_kidney,label_gallbladder,label_esophagus,label_liver,label_stomach,label_aorta,label_inferior_vena_cava,label_portal_vein_splenic_vein,label_pancreas,label_right_adrenal_gland,label_left_adrenal_gland=process_label(label)
        
        
        infer_spleen,infer_right_kidney,infer_left_kidney,infer_gallbladder,infer_esophagus,infer_liver,infer_stomach,infer_aorta,infer_inferior_vena_cava,infer_portal_vein_splenic_vein,infer_pancreas,infer_right_adrenal_gland,infer_left_adrenal_gland=process_label(infer)
        
        #Dice_spleen.append(dice(infer_spleen,label_spleen))
        #Dice_right_kidney.append(dice(infer_right_kidney,label_right_kidney))
        #Dice_left_kidney.append(dice(infer_left_kidney,label_left_kidney))
        #Dice_gallbladder.append(dice(infer_gallbladder,label_gallbladder))
        #Dice_esophagus.append(dice(infer_esophagus,label_esophagus))
        #Dice_liver.append(dice(infer_liver,label_liver))
        #Dice_stomach.append(dice(infer_stomach,label_stomach))
        #Dice_aorta.append(dice(infer_aorta,label_aorta))
        #Dice_inferior_vena_cava.append(dice(infer_inferior_vena_cava,label_inferior_vena_cava))
        #Dice_portal_vein_splenic_vein.append(dice(infer_portal_vein_splenic_vein,label_portal_vein_splenic_vein))
        #Dice_pancreas.append(dice(infer_pancreas,label_pancreas))
        #Dice_right_adrenal_gland.append(dice(infer_right_adrenal_gland,label_right_adrenal_gland))
        #Dice_left_adrenal_gland.append(dice(infer_left_adrenal_gland,label_left_adrenal_gland))
        
        hd_spleen.append(hd(infer_spleen,label_spleen))
        hd_right_kidney.append(hd(infer_right_kidney,label_right_kidney))
        hd_left_kidney.append(hd(infer_left_kidney,label_left_kidney))
        hd_gallbladder.append(hd(infer_gallbladder,label_gallbladder))
        #hd_esophagus.append(hd(infer_esophagus,label_esophagus))
        hd_liver.append(hd(infer_liver,label_liver))
        hd_stomach.append(hd(infer_stomach,label_stomach))
        hd_aorta.append(hd(infer_aorta,label_aorta))
        #hd_inferior_vena_cava.append(hd(infer_inferior_vena_cava,label_inferior_vena_cava))
        #hd_portal_vein_splenic_vein.append(hd(infer_portal_vein_splenic_vein,label_portal_vein_splenic_vein))
        hd_pancreas.append(hd(infer_pancreas,label_pancreas))
        #hd_right_adrenal_gland.append(hd(infer_right_adrenal_gland,label_right_adrenal_gland))
        #hd_left_adrenal_gland.append(hd(infer_left_adrenal_gland,label_left_adrenal_gland))
        
        
        #fw.write('*'*20+'\n',)
        #fw.write(infer_path.split('/')[-1]+'\n')
        #fw.write('Dice_spleen: {:.4f}\n'.format(Dice_spleen[-1]))
        #fw.write('Dice_right_kidney: {:.4f}\n'.format(Dice_right_kidney[-1]))
        #fw.write('Dice_left_kidney: {:.4f}\n'.format(Dice_left_kidney[-1]))
        #fw.write('Dice_gallbladder: {:.4f}\n'.format(Dice_gallbladder[-1]))
        #fw.write('Dice_esophagus: {:.4f}\n'.format(Dice_esophagus[-1]))
        #fw.write('Dice_liver: {:.4f}\n'.format(Dice_liver[-1]))
        #fw.write('Dice_stomach: {:.4f}\n'.format(Dice_stomach[-1]))
        #fw.write('Dice_aorta: {:.4f}\n'.format(Dice_aorta[-1]))
        #fw.write('Dice_inferior_vena_cava: {:.4f}\n'.format(Dice_inferior_vena_cava[-1]))
        #fw.write('Dice_portal_vein_splenic_vein: {:.4f}\n'.format(Dice_portal_vein_splenic_vein[-1]))
        #fw.write('Dice_pancreas: {:.4f}\n'.format(Dice_pancreas[-1]))
        #fw.write('Dice_right_adrenal_gland: {:.4f}\n'.format(Dice_right_adrenal_gland[-1]))
        #fw.write('Dice_left_adrenal_gland: {:.4f}\n'.format(Dice_left_adrenal_gland[-1]))
        #fw.write('*'*20+'\n')
        
        fw.write('*'*20+'\n',)
        fw.write(infer_path.split('/')[-1]+'\n')
        fw.write('Dice_spleen: {:.4f}\n'.format(hd_spleen[-1]))
        fw.write('Dice_right_kidney: {:.4f}\n'.format(hd_right_kidney[-1]))
        fw.write('Dice_left_kidney: {:.4f}\n'.format(hd_left_kidney[-1]))
        fw.write('Dice_gallbladder: {:.4f}\n'.format(hd_gallbladder[-1]))
        #fw.write('Dice_esophagus: {:.4f}\n'.format(hd_esophagus[-1]))
        fw.write('Dice_liver: {:.4f}\n'.format(hd_liver[-1]))
        fw.write('Dice_stomach: {:.4f}\n'.format(hd_stomach[-1]))
        fw.write('Dice_aorta: {:.4f}\n'.format(hd_aorta[-1]))
        #fw.write('Dice_inferior_vena_cava: {:.4f}\n'.format(hd_inferior_vena_cava[-1]))
        #fw.write('Dice_portal_vein_splenic_vein: {:.4f}\n'.format(hd_portal_vein_splenic_vein[-1]))
        fw.write('Dice_pancreas: {:.4f}\n'.format(hd_pancreas[-1]))
        #fw.write('Dice_right_adrenal_gland: {:.4f}\n'.format(hd_right_adrenal_gland[-1]))
        #fw.write('Dice_left_adrenal_gland: {:.4f}\n'.format(hd_left_adrenal_gland[-1]))
        fw.write('*'*20+'\n')
    
    #fw.write('*'*20+'\n')
    #fw.write('Mean_Dice\n')
    #fw.write('Dice_spleen'+str(np.mean(Dice_spleen))+'\n')
    #fw.write('Dice_right_kidney'+str(np.mean(Dice_right_kidney))+'\n')
    #fw.write('Dice_left_kidney'+str(np.mean(Dice_left_kidney))+'\n')
    #fw.write('Dice_gallbladder'+str(np.mean(Dice_gallbladder))+'\n')
    #fw.write('Dice_esophagus'+str(np.mean(Dice_esophagus))+'\n')
    #fw.write('Dice_liver'+str(np.mean(Dice_liver))+'\n')
    #fw.write('Dice_stomach'+str(np.mean(Dice_stomach))+'\n')
    #fw.write('Dice_aorta'+str(np.mean(Dice_aorta))+'\n')
    #fw.write('Dice_inferior_vena_cava'+str(np.mean(Dice_inferior_vena_cava))+'\n')
    #fw.write('Dice_portal_vein_splenic_vein'+str(np.mean(Dice_portal_vein_splenic_vein))+'\n')
    #fw.write('Dice_pancreas'+str(np.mean(Dice_pancreas))+'\n')
    #fw.write('Dice_right_adrenal_gland'+str(np.mean(Dice_right_adrenal_gland))+'\n')
    #fw.write('Dice_left_adrenal_gland'+str(np.mean(Dice_left_adrenal_gland))+'\n')
    #fw.write('*'*20+'\n')
    
    fw.write('*'*20+'\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_spleen'+str(np.mean(hd_spleen))+'\n')
    fw.write('Dice_right_kidney'+str(np.mean(hd_right_kidney))+'\n')
    fw.write('Dice_left_kidney'+str(np.mean(hd_left_kidney))+'\n')
    fw.write('Dice_gallbladder'+str(np.mean(hd_gallbladder))+'\n')
    #fw.write('Dice_esophagus'+str(np.mean(hd_esophagus))+'\n')
    fw.write('Dice_liver'+str(np.mean(hd_liver))+'\n')
    fw.write('Dice_stomach'+str(np.mean(hd_stomach))+'\n')
    fw.write('Dice_aorta'+str(np.mean(hd_aorta))+'\n')
    #fw.write('Dice_inferior_vena_cava'+str(np.mean(hd_inferior_vena_cava))+'\n')
    #fw.write('Dice_portal_vein_splenic_vein'+str(np.mean(hd_portal_vein_splenic_vein))+'\n')
    fw.write('Dice_pancreas'+str(np.mean(hd_pancreas))+'\n')
    #fw.write('Dice_right_adrenal_gland'+str(np.mean(hd_right_adrenal_gland))+'\n')
    #fw.write('Dice_left_adrenal_gland'+str(np.mean(hd_left_adrenal_gland))+'\n')
    fw.write('*'*20+'\n')
    
    hd=[]
    dsc=[]
    #dsc.append(np.mean(Dice_spleen))
    #dsc.append(np.mean(Dice_right_kidney))
    #dsc.append(np.mean(Dice_left_kidney))
    #dsc.append(np.mean(Dice_gallbladder))
    #dsc.append(np.mean(Dice_liver))
    #dsc.append(np.mean(Dice_stomach))
    #dsc.append(np.mean(Dice_aorta))
    #dsc.append(np.mean(Dice_pancreas))
    #fw.write('DSC:'+str(np.mean(dsc))+'\n')
    
    hd.append(np.mean(hd_spleen))
    hd.append(np.mean(hd_right_kidney))
    hd.append(np.mean(hd_left_kidney))
    hd.append(np.mean(hd_gallbladder))
    hd.append(np.mean(hd_liver))
    hd.append(np.mean(hd_stomach))
    hd.append(np.mean(hd_aorta))
    hd.append(np.mean(hd_pancreas))
    fw.write('DSC:'+str(np.mean(hd))+'\n')
    print('done')

if __name__ == '__main__':
    fold='swin_l_gelunorm'
    test(fold)
