import glob
import os
import SimpleITK as sitk
import numpy as np
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
    path='../DATASET/nnFormer_raw/nnFormer_raw_data/Task002_Synapse/'
    label_list=sorted(glob.glob(os.path.join(path,'labelsTs','*nii.gz')))
    infer_list=sorted(glob.glob(os.path.join(path,'inferTs',fold,'*nii.gz')))
    print("loading success...")
    
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
    
    file=path + 'inferTs/'+fold
    if not os.path.exists(file):
        os.makedirs(file)
    fw = open(file+'/8dice_pre.txt', 'a')
    
    for label_path,infer_path in zip(label_list,infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label,infer = read_nii(label_path),read_nii(infer_path)
        label_spleen,label_right_kidney,label_left_kidney,label_gallbladder,label_esophagus,label_liver,label_stomach,label_aorta,label_inferior_vena_cava,label_portal_vein_splenic_vein,label_pancreas,label_right_adrenal_gland,label_left_adrenal_gland=process_label(label)
        
        
        infer_spleen,infer_right_kidney,infer_left_kidney,infer_gallbladder,infer_esophagus,infer_liver,infer_stomach,infer_aorta,infer_inferior_vena_cava,infer_portal_vein_splenic_vein,infer_pancreas,infer_right_adrenal_gland,infer_left_adrenal_gland=process_label(infer)
        
        Dice_spleen.append(dice(infer_spleen,label_spleen))
        Dice_right_kidney.append(dice(infer_right_kidney,label_right_kidney))
        Dice_left_kidney.append(dice(infer_left_kidney,label_left_kidney))
        Dice_gallbladder.append(dice(infer_gallbladder,label_gallbladder))
        Dice_esophagus.append(dice(infer_esophagus,label_esophagus))
        Dice_liver.append(dice(infer_liver,label_liver))
        Dice_stomach.append(dice(infer_stomach,label_stomach))
        Dice_aorta.append(dice(infer_aorta,label_aorta))
        Dice_inferior_vena_cava.append(dice(infer_inferior_vena_cava,label_inferior_vena_cava))
        Dice_portal_vein_splenic_vein.append(dice(infer_portal_vein_splenic_vein,label_portal_vein_splenic_vein))
        Dice_pancreas.append(dice(infer_pancreas,label_pancreas))
        Dice_right_adrenal_gland.append(dice(infer_right_adrenal_gland,label_right_adrenal_gland))
        Dice_left_adrenal_gland.append(dice(infer_left_adrenal_gland,label_left_adrenal_gland))
        
        fw.write('*'*20+'\n',)
        fw.write(infer_path.split('/')[-1]+'\n')
        fw.write('Dice_spleen: {:.4f}\n'.format(Dice_spleen[-1]))
        fw.write('Dice_right_kidney: {:.4f}\n'.format(Dice_right_kidney[-1]))
        fw.write('Dice_left_kidney: {:.4f}\n'.format(Dice_left_kidney[-1]))
        fw.write('Dice_gallbladder: {:.4f}\n'.format(Dice_gallbladder[-1]))
        fw.write('Dice_esophagus: {:.4f}\n'.format(Dice_esophagus[-1]))
        fw.write('Dice_liver: {:.4f}\n'.format(Dice_liver[-1]))
        fw.write('Dice_stomach: {:.4f}\n'.format(Dice_stomach[-1]))
        fw.write('Dice_aorta: {:.4f}\n'.format(Dice_aorta[-1]))
        fw.write('Dice_inferior_vena_cava: {:.4f}\n'.format(Dice_inferior_vena_cava[-1]))
        fw.write('Dice_portal_vein_splenic_vein: {:.4f}\n'.format(Dice_portal_vein_splenic_vein[-1]))
        fw.write('Dice_pancreas: {:.4f}\n'.format(Dice_pancreas[-1]))
        fw.write('Dice_right_adrenal_gland: {:.4f}\n'.format(Dice_right_adrenal_gland[-1]))
        fw.write('Dice_left_adrenal_gland: {:.4f}\n'.format(Dice_left_adrenal_gland[-1]))
        fw.write('*'*20+'\n')
    
    fw.write('*'*20+'\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_spleen'+str(np.mean(Dice_spleen))+'\n')
    fw.write('Dice_right_kidney'+str(np.mean(Dice_right_kidney))+'\n')
    fw.write('Dice_left_kidney'+str(np.mean(Dice_left_kidney))+'\n')
    fw.write('Dice_gallbladder'+str(np.mean(Dice_gallbladder))+'\n')
    fw.write('Dice_esophagus'+str(np.mean(Dice_esophagus))+'\n')
    fw.write('Dice_liver'+str(np.mean(Dice_liver))+'\n')
    fw.write('Dice_stomach'+str(np.mean(Dice_stomach))+'\n')
    fw.write('Dice_aorta'+str(np.mean(Dice_aorta))+'\n')
    fw.write('Dice_inferior_vena_cava'+str(np.mean(Dice_inferior_vena_cava))+'\n')
    fw.write('Dice_portal_vein_splenic_vein'+str(np.mean(Dice_portal_vein_splenic_vein))+'\n')
    fw.write('Dice_pancreas'+str(np.mean(Dice_pancreas))+'\n')
    fw.write('Dice_right_adrenal_gland'+str(np.mean(Dice_right_adrenal_gland))+'\n')
    fw.write('Dice_left_adrenal_gland'+str(np.mean(Dice_left_adrenal_gland))+'\n')
    fw.write('*'*20+'\n')
    
    dsc=[]
    dsc.append(np.mean(Dice_spleen))
    dsc.append(np.mean(Dice_right_kidney))
    dsc.append(np.mean(Dice_left_kidney))
    dsc.append(np.mean(Dice_gallbladder))
    #dsc.append(np.mean(Dice_esophagus))
    dsc.append(np.mean(Dice_liver))
    dsc.append(np.mean(Dice_stomach))
    dsc.append(np.mean(Dice_aorta))
    #dsc.append(np.mean(Dice_inferior_vena_cava))
    #dsc.append(np.mean(Dice_portal_vein_splenic_vein))
    dsc.append(np.mean(Dice_pancreas))
    #dsc.append(np.mean(Dice_right_adrenal_gland))
    #dsc.append(np.mean(Dice_left_adrenal_gland))
    fw.write('DSC:'+str(np.mean(dsc))+'\n')
    print('done')

if __name__ == '__main__':
    fold='output'
    test(fold)
