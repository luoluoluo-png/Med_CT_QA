# Med_CT_QA
Med_CT_QA
这是给CT质量评估的代码
Install Requirements
pytorch
opencv
scipy
pandas
torchvision
torchvideo

Train models

1.    Extract key frames (Set the file path internally)

python extract_key_frames.py

2.    Extract brightness consistency features

python brightness_consistency.py

3.    Extract temporal features

python extract_temporal_features.py

4.    Train the model

python train.py

5.    Test the model

python test.py



iqa_3D_sh_4_multi.py：对dcm序列文件夹进行预测，分别预测旋转角度，缺失比，运动伪影，金属伪影，并且保存结果到txt文档（输入路径下就是各个病人的文件夹，可以遍历每一位病人）
iqa_3D_sh_5_multi.py:对dcm序列文件夹进行预测，分别预测旋转角度，缺失比，mos分数，运动伪影，金属伪影，并且保存结果到txt文档（输入路径下就是各个病人的文件夹，可以遍历每一位病人）
