import glob
import os
import SimpleITK as sitk
import numpy as np
from medpy.metric import binary
from sklearn.neighbors import KDTree
from scipy import ndimage


def read_nii(path):
    itk_img=sitk.ReadImage(path)
    spacing=np.array(itk_img.GetSpacing())
    return sitk.GetArrayFromImage(itk_img),spacing

def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())

def process_label(label):
    rv = label == 1
    myo = label == 2
    lv = label == 3
    
    return rv,myo,lv
'''    
def hd(pred,gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = binary.dc(pred, gt)
        hd95 = binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0
'''

def hd(pred,gt):
    #labelPred=sitk.GetImageFromArray(lP.astype(np.float32), isVector=False)
    #labelTrue=sitk.GetImageFromArray(lT.astype(np.float32), isVector=False)
    #hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    #hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    #return hausdorffcomputer.GetAverageHausdorffDistance()
    if pred.sum() > 0 and gt.sum()>0:
        hd95 = binary.hd95(pred, gt)
        print(hd95)
        return  hd95
    else:
        return 0






def test(fold):
    #path='/home/xychen/new_transformer/nnUNetFrame/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task002_Abdomen/nnUNetTrainerV2_swin_l_mix_head3__nnUNetPlansv2.1/fold_0/validation_raw/'
    path='/home/xychen/new_transformer/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task009_ACDC/'
    label_list=sorted(glob.glob(os.path.join(path,'labelsTs','*nii.gz')))
    infer_list=sorted(glob.glob(os.path.join(path,'inferTs',fold,'*nii.gz')))
    #infer_list =sorted(glob.glob('/home/xychen/new_transformer/nnUNetFrame/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task002_Abdomen/nnUNetTrainerV2_swin_l_multi_head__nnUNetPlansv2.1/fold_0/validation_raw/*.nii.gz'))
    print("loading success...")
    print(label_list)
    print(infer_list)
    Dice_rv=[]
    Dice_myo=[]
    Dice_lv=[]
    
    hd_rv=[]
    hd_myo=[]
    hd_lv=[]
    
    file=path + 'inferTs/'+fold
    if not os.path.exists(file):
        os.makedirs(file)
    fw = open(file+'/dice_pre.txt', 'w')
    
    for label_path,infer_path in zip(label_list,infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label,spacing= read_nii(label_path)
        infer,spacing= read_nii(infer_path)
        label_rv,label_myo,label_lv=process_label(label)
        infer_rv,infer_myo,infer_lv=process_label(infer)
        
        Dice_rv.append(dice(infer_rv,label_rv))
        Dice_myo.append(dice(infer_myo,label_myo))
        Dice_lv.append(dice(infer_lv,label_lv))
        
        hd_rv.append(hd(infer_rv,label_rv))
        hd_myo.append(hd(infer_myo,label_myo))
        hd_lv.append(hd(infer_lv,label_lv))
        
        fw.write('*'*20+'\n',)
        fw.write(infer_path.split('/')[-1]+'\n')
        #fw.write('hd_rv: {:.4f}\n'.format(hd_rv[-1]))
        #fw.write('hd_myo: {:.4f}\n'.format(hd_myo[-1]))
        #fw.write('hd_lv: {:.4f}\n'.format(hd_lv[-1]))
        #fw.write('*'*20+'\n')
        fw.write('*'*20+'\n',)
        fw.write(infer_path.split('/')[-1]+'\n')
        fw.write('Dice_rv: {:.4f}\n'.format(Dice_rv[-1]))
        fw.write('Dice_myo: {:.4f}\n'.format(Dice_myo[-1]))
        fw.write('Dice_lv: {:.4f}\n'.format(Dice_lv[-1]))
        fw.write('*'*20+'\n')
        
    #fw.write('*'*20+'\n')
    #fw.write('Mean_hd\n')
    #fw.write('hd_rv'+str(np.mean(hd_rv))+'\n')
    #fw.write('hd_myo'+str(np.mean(hd_myo))+'\n')
    #fw.write('hd_lv'+str(np.mean(hd_lv))+'\n')   
    #fw.write('*'*20+'\n')
    
    fw.write('*'*20+'\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_rv'+str(np.mean(Dice_rv))+'\n')
    fw.write('Dice_myo'+str(np.mean(Dice_myo))+'\n')
    fw.write('Dice_lv'+str(np.mean(Dice_lv))+'\n')   
    fw.write('*'*20+'\n')
    
    dsc=[]
    dsc.append(np.mean(Dice_rv))
    dsc.append(np.mean(Dice_myo))
    dsc.append(np.mean(Dice_lv))
    #avg_hd=[]
    #avg_hd.append(np.mean(hd_rv))
    #avg_hd.append(np.mean(hd_myo))
    #avg_hd.append(np.mean(hd_lv))
    #fw.write('avg_hd:'+str(np.mean(avg_hd))+'\n')

    fw.write('DSC:'+str(np.mean(dsc))+'\n')
    print('done')

if __name__ == '__main__':
    fold='time_mix'
    test(fold)
