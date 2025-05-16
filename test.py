import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from scipy.stats import spearmanr, pearsonr
from my_dataloader_test import DicomDataset_spatio_temporal_brightness
from final_fusion_model import swin_small_patch4_window7_224 as create_model
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

# 权重文件路径
WEIGHTS_PATH = '/home/gem/sh/new_light_vqa/512_size/output/4_6/last2_SI+TI_epoch_39_avg_auc_0.768264.pth'
# 数据路径
DCM_PATH = '/mnt/gemlab_data_2/User_database/shenhao/whole_sz_and_luna16/whole_sz_and_luna16_data'
VIDEOS_DIR = '/mnt/gemlab_data_2/User_database/shenhao/whole_sz_and_luna16/key_frames_2_21'
DATA_DIR_3D = '/mnt/gemlab_data_2/User_database/shenhao/whole_sz_and_luna16/temporal_feature2_21/003493'
BRIGHTNESS = '/mnt/gemlab_data_2/User_database/shenhao/whole_sz_and_luna16/brightness_consistency_2_21'
DATAINFO_TEST = '/home/gem/sh/new_light_vqa/512_size/my_test_true_balance_data.csv'

    # videos_dir = '/mnt/gemlab_data_2/User_database/shenhao/Light_MQA_master/key_frames_11_11'
    # data_dir_3D = '/mnt/gemlab_data_2/User_database/shenhao/Light_MQA_master/temporal_feature11_11'
    # brightness = '/mnt/gemlab_data_2/User_database/shenhao/Light_MQA_master/brightness_consistency_11_11'
    # dcm_path='/mnt/gemlab_data_2/User_database/shenhao/whole_sz_data'
    # datainfo_train = 'my_train.csv'
    # datainfo_test = 'my_val.csv'

def evaluate(model, test_loader, device):
    model.eval()
    targets1 = []
    predictions1 = []
    targets2 = []
    predictions2 = []
    targets3 = []
    predictions3 = []
    targets4 = []
    predictions4 = []

    with torch.no_grad():
        for i, (video, tem_f, _, target1, target2, target3, target4) in enumerate(test_loader):
            video = video.to(device)
            tem_f = tem_f.to(device)
            video = torch.reshape(video, [video.shape[0] * video.shape[1], 3, 512, 512])
            tem_f = torch.reshape(tem_f, [tem_f.shape[0] * tem_f.shape[1], 2304 + 144])

            target1 = target1.to(device, dtype=torch.float32)
            target2 = target2.to(device, dtype=torch.float32)
            #target3 = target3.to(device, dtype=torch.float32)
            target3 = target3.to(device, dtype=torch.long)
            target4 = target4.to(device, dtype=torch.long)

            outputs = model(video, tem_f)

            targets1.extend(target1.cpu().numpy().reshape(-1))
            predictions1.extend(torch.mean(outputs[0]).detach().cpu().numpy().reshape(-1))
            targets2.extend(target2.cpu().numpy().reshape(-1))
            predictions2.extend(torch.mean(outputs[1]).detach().cpu().numpy().reshape(-1))
            # targets3.extend(target3.cpu().numpy().reshape(-1))
            # predictions3.extend(torch.mean(outputs[2]).detach().cpu().numpy().reshape(-1))
            targets3.extend(target3.cpu().numpy())
            predictions3.append(torch.mean(outputs[2], dim=0).detach().cpu().numpy())
            targets4.extend(target4.cpu().numpy())
            predictions4.append(torch.mean(outputs[3], dim=0).detach().cpu().numpy())
    print(predictions3,'这是预测的prediction3')
    print(target3,'这是target3')
    print(predictions4,'这是预测的prediction4')
    print(target4,'这是target4')
    predictions4_np = np.array(predictions4)
    predictions3_np = np.array(predictions3)
    threshold = 0.5
    predictions4_labels = (predictions4_np[:, 1] > threshold).astype(int)
    predictions3_labels = (predictions3_np[:, 1] > threshold).astype(int)
    val_srcc1 = spearmanr(targets1, predictions1)[0]
    val_plcc1 = pearsonr(targets1, predictions1)[0]
    val_rmse1 = mean_squared_error(targets1, predictions1, squared=False)
    val_srcc2 = spearmanr(targets2, predictions2)[0]
    val_plcc2 = pearsonr(targets2, predictions2)[0]
    val_rmse2 = mean_squared_error(targets2, predictions2, squared=False)
    # val_srcc3 = spearmanr(targets3, predictions3)[0]
    # val_plcc3 = pearsonr(targets3, predictions3)[0]
    # val_rmse3 = mean_squared_error(targets3, predictions3, squared=False)
    
    val_accuracy3 = accuracy_score(targets3, predictions3_labels)
    val_f1_3 = f1_score(targets3, predictions3_labels)
    val_recall3 = recall_score(targets3, predictions3_labels)
    val_precision3 = precision_score(targets3, predictions3_labels)
    
    val_accuracy4 = accuracy_score(targets4, predictions4_labels)
    val_f1_4 = f1_score(targets4, predictions4_labels)
    val_recall4 = recall_score(targets4, predictions4_labels)
    val_precision4 = precision_score(targets4, predictions4_labels)

    print(predictions3_labels,'这是预测的predictions3_labels')
    print(targets3,'这是targets3')
    print(predictions4_labels,'这是预测的predictions4_labels')
    print(targets4,'这是targets4')
    
    print(f'Test Results:\n'
          f'SRCC1: {val_srcc1:.4f}, PLCC1: {val_plcc1:.4f}, RMSE1: {val_rmse1:.4f}\n'
          f'SRCC2: {val_srcc2:.4f}, PLCC2: {val_plcc2:.4f}, RMSE2: {val_rmse2:.4f}\n'
          f'Accuracy_3: {val_accuracy3:.4f}, F1_3: {val_f1_3:.4f}, Recall_3: {val_recall3:.4f}, Precision_3: {val_precision3:.4f}, AUC_3: {roc_auc_score(targets3, predictions3_np[:, 1]):.4f}\n'
          f'Accuracy_4: {val_accuracy4:.4f}, F1_4: {val_f1_4:.4f}, Recall_4: {val_recall4:.4f}, Precision_4: {val_precision4:.4f}, AUC_4: {roc_auc_score(targets4, predictions4_np[:, 1]):.4f}')

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)

    # 加载模型权重
    assert os.path.exists(WEIGHTS_PATH), f"Weights file: '{WEIGHTS_PATH}' does not exist."
    weights_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(weights_dict, strict=False)

    transformations_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    testset = DicomDataset_spatio_temporal_brightness('test', DCM_PATH, VIDEOS_DIR, DATA_DIR_3D, BRIGHTNESS, DATAINFO_TEST, transformations_test, 'my_test', 'SlowFast')

    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=4)

    evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()
