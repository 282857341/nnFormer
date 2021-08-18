%安装nnunet
%git clone https://github.com/MIC-DKFZ/nnUNet.git
%cd nnUNet
%pip install -e .

数据集格式
https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md

新建文件夹
数据集放在./Dataset/nnUNet_raw/nnUNet_raw_data/
./Dataset/nnUNet_raw/nnUNet_raw_data/Task01_ACDC/imagesTr/
./Dataset/nnUNet_raw/nnUNet_raw_data/Task01_ACDC/imagesTs/
./Dataset/nnUNet_raw/nnUNet_raw_data/Task01_ACDC/labelsTr/
./Dataset/nnUNet_raw/nnUNet_raw_data/Task01_ACDC/labelsTs/
./Dataset/nnUNet_raw/nnUNet_raw_data/Task02_Synapse/imagesTr/
./Dataset/nnUNet_raw/nnUNet_raw_data/Task02_Synapse/imagesTs/
./Dataset/nnUNet_raw/nnUNet_raw_data/Task02_Synapse/labelsTr/
./Dataset/nnUNet_raw/nnUNet_raw_data/Task02_Synapse/labelsTs/
./Dataset/nnUNet_raw/nnUNet_cropped_data/
./Dataset/nnUNet_trained_models
./Dataset/nnUNet_preprocessed


生成./Dataset/nnUNet_raw/nnUNet_raw_data/Task01_ACDC/dataset.json
生成./Dataset/nnUNet_raw/nnUNet_raw_data/Task02_Synapse/dataset.json

转换数据格式
python ./experiment_planning/nnUNet_convert_decathlon_task.py -i ./Dataset/nnUNet_raw/nnUNet_raw_data/Task01_ACDC
python ./experiment_planning/nnUNet_convert_decathlon_task.py -i ./Dataset/nnUNet_raw/nnUNet_raw_data/Task02_Synapse
在这个Task01_ACDC文件夹旁边会多了一个Task001_ACDC
#在/home/.bashrc内添加下述内容后，输入source .bashrc 更新：
#export nnUNet_raw_data_base="/home/hostname/nnUNetFrame/DATASET/nnUNet_raw"
#export nnUNet_preprocessed="/home/hostname/nnUNetFrame/DATASET/nnUNet_preprocessed"
#export RESULTS_FOLDER="/home/hostname/nnUNetFrame/DATASET/nnUNet_trained_models

预处理数据
python ./experiment_planning/nnUNet_plan_and_preprocess.py -t 1
python ./experiment_planning/nnUNet_plan_and_preprocess.py -t 2
python ./change_plan_swin.py -t 1
python ./change_plan_swin.py -t 2

——————————————————————————————————————————————————————————————



推理我们已训练好的模型
把下载好的模型文件放在
./Dataset/nnUNet_trained_models/nnUNet/3d_fullres/Task001_ACDC/nnUNetTrainerV2_ACDC__nnUNetPlansv2.1/model_best.model
./Dataset/nnUNet_trained_models/nnUNet/3d_fullres/Task002_Synapse/nnUNetTrainerV2_Synapse__nnUNetPlansv2.1/model_best.model

推理：
python ./inference/predict_simple.py -i ./Dataset/nnUNet_raw/nnUNet_raw_data/Task001_ACDC/imagesTs -o ./Dataset/nnUNet_raw/nnUNet_raw_data/Task001_ACDC/inferTs/output -m 3d_fullres -f 0 -t 1 -chk model_best -tr nnUNetTrainerV2_ACDC

python ./inference/predict_simple.py -i ./Dataset/nnUNet_raw/nnUNet_raw_data/Task002_Synapse/imagesTs -o ./Dataset/nnUNet_raw/nnUNet_raw_data/Task002_Synapse/inferTs/output -m 3d_fullres -f 0 -t 2 -chk model_best -tr nnUNetTrainerV2_Synapse

#测试集的推理结果会保存在 ./inferTs/output/内

计算dice：
python ./ACDC_dice/inference.py
python ./Synapse_dice_and_hd/inference.py
dice结果会保存在./infer/output内


——————————————————————————————————————————————————————————————

若执行了推理已训练的模型，需要对./inferTs/output/的output文件夹改名，以免影响后续推理
train from scratch（从头训练）
新建文件夹
./Pretrained_weight/

把下载好的预训练权重放在里面
./Pretrained_weight/pretrain_ACDC.model
./Pretrained_weight/pretrain_Synapse.model

模型训练命令：
python ./run/run_training.py 3d_fullres nnUNetTrainerV2_ACDC 1 0 
python ./run/run_training.py 3d_fullres nnUNetTrainerV2_Synapse 2 0 

推理：
python ./inference/predict_simple.py -i ./Dataset/nnUNet_raw/nnUNet_raw_data/Task001_ACDC/imagesTs -o ./Dataset/nnUNet_raw/nnUNet_raw_data/Task001_ACDC/inferTs/output -m 3d_fullres -f 0 -t 1 -chk model_best -tr nnUNetTrainerV2_ACDC

python ./inference/predict_simple.py -i ./Dataset/nnUNet_raw/nnUNet_raw_data/Task002_Synapse/imagesTs -o ./Dataset/nnUNet_raw/nnUNet_raw_data/Task002_Synapse/inferTs/output -m 3d_fullres -f 0 -t 2 -chk model_best -tr nnUNetTrainerV2_Synapse

#测试集的推理结果会保存在 ./inferTs/output/内

计算dice：
python ./ACDC_dice/inference.py
python ./Synapse_dice_and_hd/inference.py
dice结果会保存在./infer/output内





