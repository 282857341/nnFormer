安装nnunet
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .

在/home/.bashrc内添加下述内容后，输入source .bashrc 更新：
export nnUNet_raw_data_base="/home/hostname/nnUNetFrame/DATASET/nnUNet_raw"
export nnUNet_preprocessed="/home/hostname/nnUNetFrame/DATASET/nnUNet_preprocessed"
export RESULTS_FOLDER="/home/hostname/nnUNetFrame/DATASET/nnUNet_trained_models

数据集格式
https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md
(数据命名：Task001_ACDC、Task002_Synapse)
预处理数据
nnUNet_plan_and_preprocess -t XXX

放入我们的网络结构到：
xxx/nnunet/network_architecture

放入我们的训练器到：
xxx/nnunet/training/network_training

模型训练命令：
nnUNet_train 3d_fullres nnUNetTrainerV2_ACDC 1 0 
nnUNet_train 3d_fullres nnUNetTrainerV2_Synapse 2 0 

推理：
cd ./Task001_ACDC #这一步是转换到数据集目录，方便后续的推理
nnUNet_predict -i imagesTs -o inferTs/output -m 3d_fullres -f 0 -t 1 -chk model_best -tr nnUNetTrainerV2_ACDC

cd ./Task002_Synapse #这一步是转换到数据集目录，方便后续的推理
nnUNet_predict -i imagesTs -o inferTs/output -m 3d_fullres -f 0 -t 2 -chk model_best -tr nnUNetTrainerV2_Synapse

测试集的推理结果会保存在 ./inferTs/output/内

计算dice：
python inference.py
dice结果会保存在./infer/output内



