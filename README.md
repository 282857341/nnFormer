# nnFormer: Interleaved Transformer for Volumetric Segmentation 

Code for paper "nnFormer: Interleaved Transformer for Volumetric Segmentation ". Please read our preprint at the following link: [paper_address](https://arxiv.org/abs/2109.03201).

Parts of codes are borrowed from [nn-UNet](https://github.com/MIC-DKFZ/nnUNet).

---
## Installation
#### 1、System requirements
This software was originally designed and run on a system running Ubuntu 18.01, with Python 3.6, PyTorch 1.8.1, and CUDA 10.1. For a full list of software packages and version numbers, see the Conda environment file `environment.yml`. 

This software leverages graphical processing units (GPUs) to accelerate neural network training and evaluation; systems lacking a suitable GPU will likely take an extremely long time to train or evaluate models. The software was tested with the NVIDIA RTX 2080 TI GPU, though we anticipate that other GPUs will also work, provided that the unit offers sufficient memory. 

#### 2、Installation guide
We recommend installation of the required packages using the Conda package manager, available through the Anaconda Python distribution. Anaconda is available free of charge for non-commercial use through [Anaconda Inc](https://www.anaconda.com/products/individual). After installing Anaconda and cloning this repository, For use as integrative framework：
```
git clone https://github.com/282857341/nnFormer.git
cd nnFormer
conda env create -f environment.yml
source activate nnFormer
pip install -e .
```

#### 3、The main downloaded file directory description 
- Evaluate:

  - nnFormer/nnformer/inference_acdc.py
  
  - nnFormer/nnformer/inference_synapse.py
  
  - nnFormer/nnformer/inference_tumor.py

- Data split:

  - nnFormer/nnformer/dataset_json/
  
- inference:

  - nnFormer/nnformer/inference/predict_simple.py

- network_architecture:

  - nnFormer/nnformer/network_architecture/nnFormer_acdc.py
  
  - nnFormer/nnformer/network_architecture/nnFormer_synapse.py.py
  
  - nnFormer/nnformer/network_architecture/nnFormer_tumor.py.py

- train:

  - nnFormer/nnformer/run/run_training.py
 
- trainer:

  - nnFormer/nnformer/training/network_training/nnFormerTrainerV2_nnformer_acdc.py

  - nnFormer/nnformer/training/network_training/nnFormerTrainerV2_nnformer_synapse.py.py
  
  - nnFormer/nnformer/training/network_training/nnFormerTrainerV2_nnformer_tumor.py.py
---

## Training
#### 1、Datasets
Datasets can be downloaded at the following links:

And the division of the dataset can be seen in the files in the ./dataset_json/

**Dataset I**
[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

**Dataset II**
[The Synapse multi-organ CT dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)

**Dataset III**
[Brain_tumor](http://medicaldecathlon.com/)

#### 2、Setting up the datasets
While we provide code to load data for training a deep-learning model, you will first need to download images from the above repositories. Regarding the format setting and related preprocessing of the dataset, we operate based on nnFormer, so I won’t go into details here. You can see [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) for specific operations. 

Regarding the downloaded data, I will not introduce too much here, you can go to the corresponding website to view it. Organize the downloaded DataProcessed as follows:

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
You can use the dataset.json in the file nnFormer/nnformer/dataset_json/

After that, you can preprocess the data using:
```
nnFormer_convert_decathlon_task -i ../DATASET/nnFormer_raw/nnFormer_raw_data/Task01_ACDC
nnFormer_convert_decathlon_task -i ../DATASET/nnFormer_raw/nnFormer_raw_data/Task02_Synapse
nnFormer_convert_decathlon_task -i ../DATASET/nnFormer_raw/nnFormer_raw_data/Task03_tumor

nnFormer_plan_and_preprocess -t 1
nnFormer_plan_and_preprocess -t 2
nnFormer_plan_and_preprocess -t 3
```

#### 3 Training and Testing the models
- Command

```
bash train_inference.sh -c 0 -n nnformer_acdc -t 1 
#-c means the id of the cuda device
#-n means the suffix of the trainer located at nnFormer/nnformer/training/network_training/
#-t means the id of the task
```
You need to adjust the path for yourself

the inference.py is located at nnFormer/nnformer 

train_inference.sh is located at nnFormer

more detail about the command: [train](https://github.com/MIC-DKFZ/nnUNet#3d-full-resolution-u-net) and [inference](https://github.com/MIC-DKFZ/nnUNet#run-inference)
##### A. Download the best model we have trained to infer the test set

The Google Drive link is as follows：
```
soon will upload
```

Put the downloaded the best training weights in the specified directory:
```
../DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task001_ACDC/nnFormerTrainerV2_nnformer_acdc__nnFormerPlansv2.1/fold_0/model_best.model
../DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task001_ACDC/nnFormerTrainerV2_nnformer_acdc__nnFormerPlansv2.1/fold_0/model_best.model.pkl

../DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task002_Synapse/nnFormerTrainerV2_nnformer_synapse__nnFormerPlansv2.1/fold_0/model_best.model
../DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task002_Synapse/nnFormerTrainerV2_nnformer_synapse__nnFormerPlansv2.1/fold_0/model_best.model.pkl

../DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task003_tumor/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1/fold_0/model_best.model
../DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task003_tumor/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1/fold_0/model_best.model.pkl
```

##### B. Download the pretrained weights for retraining the model and inference

The Google Drive link is as follows：

```
soon will upload
```
Don't forget to change self.load_pretrain_weight in the trainer

Put the pretrain weight in the specified directory:
```
../Pretrained_weight/pretrain_Synapse.model
../Pretrained_weight/pretrain_ACDC.model
../Pretrained_weight/pretrain_tumor.model
```
