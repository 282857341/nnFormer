# nnFormer: Volumetric Medical Image Segmentation via a 3D Transformer

At 2022/02/11, we rebuilt the code of nnFormer to match the performance reported in the latest [draft](https://arxiv.org/abs/2109.03201). The results produced by new codes are more stable and thus easier to reproduce! 

---
## Installation
#### 1、System requirements
We run nnFormer on a system running Ubuntu 18.01, with Python 3.6, PyTorch 1.8.1, and CUDA 10.1. For a full list of software packages and version numbers, see the Conda environment file `environment.yml`. 

This software leverages graphical processing units (GPUs) to accelerate neural network training and evaluation. Thus, systems lacking a suitable GPU would likely take an extremely long time to train or evaluate models. The software was tested with the NVIDIA RTX 2080 TI GPU, though we anticipate that other GPUs will also work, provided that the unit offers sufficient memory. 

#### 2、Installation guide
We recommend installation of the required packages using the conda package manager, available through the Anaconda Python distribution. Anaconda is available free of charge for non-commercial use through [Anaconda Inc](https://www.anaconda.com/products/individual). After installing Anaconda and cloning this repository, For use as integrative framework：
```
git clone https://github.com/282857341/nnFormer.git
cd nnFormer
conda env create -f environment.yml
source activate nnFormer
pip install -e .
```

#### 3、Functions of scripts and folders
- **For evaluation:**
  - ``nnFormer/nnformer/inference_acdc.py``
  
  - ``nnFormer/nnformer/inference_synapse.py``
  
  - ``nnFormer/nnformer/inference_tumor.py``
  
- **Data split:**
  - ``nnFormer/nnformer/dataset_json/``
  
- **For inference:**
  - ``nnFormer/nnformer/inference/predict_simple.py``
  
- **Network architecture:**
  - ``nnFormer/nnformer/network_architecture/nnFormer_acdc.py``
  
  - ``nnFormer/nnformer/network_architecture/nnFormer_synapse.py.py``
  
  - ``nnFormer/nnformer/network_architecture/nnFormer_tumor.py.py``
  
- **For training:**
  - ``nnFormer/nnformer/run/run_training.py``
  
- **Trainer for dataset:**
  - ``nnFormer/nnformer/training/network_training/nnFormerTrainerV2_nnformer_acdc.py``
  
  - ``nnFormer/nnformer/training/network_training/nnFormerTrainerV2_nnformer_synapse.py.py``
  
  - ``nnFormer/nnformer/training/network_training/nnFormerTrainerV2_nnformer_tumor.py.py``
---

## Training
#### 1. Dataset download
Datasets can be acquired via following links:

**Dataset I**
[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

**Dataset II**
[The Synapse multi-organ CT dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)

**Dataset III**
[Brain_tumor](http://medicaldecathlon.com/)

The splits of all three datasets are available in ``nnFormer/nnformer/dataset_json/``.

#### 2. Setting up the datasets
After you have downloaded the datasets, you can follow the settings in [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) for path configurations and preprocessing procedures. Finally, your folders should be organized as follows:

```
./Pretrained_weight/
./nnFormer/
./DATASET/
  ├── nnFormer_raw/
      ├── nnFormer_raw_data/
          ├── Task01_ACDC/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task02_Synapse/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task03_tumor/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
      ├── nnFormer_cropped_data/
  ├── nnFormer_trained_models/
  ├── nnFormer_preprocessed/
```
You can refer to ``nnFormer/nnformer/dataset_json/`` for data split.

After that, you can preprocess the above data using following commands:
```
nnFormer_convert_decathlon_task -i ../DATASET/nnFormer_raw/nnFormer_raw_data/Task01_ACDC
nnFormer_convert_decathlon_task -i ../DATASET/nnFormer_raw/nnFormer_raw_data/Task02_Synapse
nnFormer_convert_decathlon_task -i ../DATASET/nnFormer_raw/nnFormer_raw_data/Task03_tumor

nnFormer_plan_and_preprocess -t 1
nnFormer_plan_and_preprocess -t 2
nnFormer_plan_and_preprocess -t 3
```

#### 3. Training and Testing
- Commands for training and testing:

```
bash train_inference.sh -c 0 -n nnformer_acdc -t 1 
#-c stands for the index of your cuda device
#-n denotes the suffix of the trainer located at nnFormer/nnformer/training/network_training/
#-t denotes the task index
```
- Donwload the pretrain weight or our best trained model:
```
Goggle Drive Link：https://drive.google.com/drive/folders/1yvqlkeRq1qr5RxH-EzFyZEFsJsGFEc78?usp=sharing
```



