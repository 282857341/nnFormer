# nnFormer: Incorporating Convolution HelpsTransformer Outperform nnU-Net in VolumetricSegmentation

Code for paper "nnFormer: Incorporating Convolution HelpsTransformer Outperform nnU-Net in VolumetricSegmentation". Please read our preprint at the following link: [paper_address](paper_address)

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
- ACDC_dice:
Calculate dice of ACDC dataset

- Synapse_dice_and_hd:
Calulate dice of the Synapse dataset

- dataset_json:
About how to divide the training and test set

- inference:
The entry program of the infernece.

- network_architecture:
The models are stored here.

- run:
The entry program of the training.

- training:
The trainers are stored here, the training of the network is conducted by the trainer.

---

## Training
#### 1、Datasets
Datasets can be downloaded at the following links:

**Dataset I**
[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

**Dataset II**
[The Synapse multi-organ CT dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)

#### 2、Setting up the datasets
While we provide code to load data for training a deep-learning model, you will first need to download images from the above repositories. Regarding the format setting and related preprocessing of the dataset, we operate based on nnUNet, so I won’t go into details here. You can see [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) for specific operations. 

Regarding the downloaded data, I will not introduce too much here, you can go to the corresponding website to view it. Organize the downloaded DataProcessed as follows:

```
./Pretrained_weight/
./nnFormer/
./DATASET/
  ├── nnUNet_raw/
      ├── nnUNet_raw_data/
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
      ├── nnUNet_cropped_data/
  ├── nnUNet_trained_models/
  ├── nnUNet_preprocessed/
```
#### 3 Generating plan files of our network
```
python ./nnunet/change_plan_swin.py 1
python ./nnunet/change_plan_swin.py 2
```
#### 4 Training and Testing the models
##### A. Use the best model we have trained to infer the test set
##### (1)、Put the downloaded the best training weights in the specified directory.
the download link is 
```
Link：https://pan.baidu.com/s/1h1h8_DKvve8enyTiIyzfHw 
Extraction code：yimv
```

the specified directory is
```
../DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task001_ACDC/nnUNetTrainerV2_ACDC__nnUNetPlansv2.1/fold_0/model_best.model
../DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task001_ACDC/nnUNetTrainerV2_ACDC__nnUNetPlansv2.1/fold_0/model_best.model.pkl

../DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task002_Synapse/nnUNetTrainerV2_Synapse__nnUNetPlansv2.1/fold_0/model_best.model
../DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task002_Synapse/nnUNetTrainerV2_Synapse__nnUNetPlansv2.1/fold_0/model_best.model.pkl
```
##### (2)、Evaluating the models
- ACDC

Inference
```
nnUNet_predict -i ../DATASET/nnUNet_raw/nnUNet_raw_data/Task001_ACDC/imagesTs -o ../DATASET/nnUNet_raw/nnUNet_raw_data/Task001_ACDC/inferTs/output -m 3d_fullres -f 0 -t 1 -chk model_best -tr nnUNetTrainerV2_ACDC
```

Calculate DICE

```
python ./nnunet/ACDC_dice/inference.py
```

- The Synapse multi-organ CT dataset

Inference
```
nnUNet_predict -i ../DATASET/nnUNet_raw/nnUNet_raw_data/Task002_Synapse/imagesTs -o ../DATASET/nnUNet_raw/nnUNet_raw_data/Task002_Synapse/inferTs/output -m 3d_fullres -f 0 -t 2 -chk model_best -tr nnUNetTrainerV2_Synapse
```
Calculate DICE
```
python ./nnunet/Synapse_dice_and_hd/inference.py
```

The dice result will be saved in ../DATASET/nnUNet_raw/nnUNet_raw_data/Task002_Synapse/inferTs/output

##### B. The complete process of retraining the model and inference
##### (1)、Put the downloaded pre-training weights in the specified directory.
the download link is 
```
Link：https://pan.baidu.com/s/1h1h8_DKvve8enyTiIyzfHw 
Extraction code：yimv
```
the specified directory is
```
../Pretrained_weight/pretrain_ACDC.model
../Pretrained_weight/pretrain_Synapse.model
```

##### (2)、Training 
- ACDC
```
nnUNet_train 3d_fullres nnUNetTrainerV2_ACDC 1 0 
```

- The Synapse multi-organ CT dataset
```
nnUNet_train 3d_fullres nnUNetTrainerV2_Synapse 2 0 
```

##### (3)、Evaluating the models
- ACDC

Inference
```
nnUNet_predict -i ../DATASET/nnUNet_raw/nnUNet_raw_data/Task001_ACDC/imagesTs -o ../DATASET/nnUNet_raw/nnUNet_raw_data/Task001_ACDC/inferTs/output -m 3d_fullres -f 0 -t 1 -chk model_best -tr nnUNetTrainerV2_ACDC
```

Calculate DICE

```
python ./nnunet/ACDC_dice/inference.py
```

- The Synapse multi-organ CT dataset

Inference
```
nnUNet_predict -i ../DATASET/nnUNet_raw/nnUNet_raw_data/Task002_Synapse/imagesTs -o ../DATASET/nnUNet_raw/nnUNet_raw_data/Task002_Synapse/inferTs/output -m 3d_fullres -f 0 -t 2 -chk model_best -tr nnUNetTrainerV2_Synapse
```
Calculate DICE
```
python ./nnunet/Synapse_dice_and_hd/inference.py
```

The dice result will be saved in ../DATASET/nnUNet_raw/nnUNet_raw_data/Task002_Synapse/inferTs/output
