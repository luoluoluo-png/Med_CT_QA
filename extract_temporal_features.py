import argparse
import os
import numpy as np
import torch
from my_dataloader import  DicomDatasetExtractFeature
from torchvision import transforms
from pytorchvideo.models.hub import slowfast_r50
import torch.nn as nn


def pack_pathway_output(frames, device):
    fast_pathway = frames
    slow_pathway = torch.index_select(
        frames,
        2,
        torch.linspace(
            0, frames.shape[2] - 1, frames.shape[2] // 4
        ).long(),
    )
    frame_list = [slow_pathway.to(device), fast_pathway.to(device)]
    return frame_list


class slowfast(torch.nn.Module):
    def __init__(self):
        super(slowfast, self).__init__()
        slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])

        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(0, 5):
            self.feature_extraction.add_module(str(x), slowfast_pretrained_features[x])

        self.slow_avg_pool.add_module('slow_avg_pool', slowfast_pretrained_features[5].pool[0])
        self.fast_avg_pool.add_module('fast_avg_pool', slowfast_pretrained_features[5].pool[1])
        self.adp_avg_pool.add_module('adp_avg_pool', slowfast_pretrained_features[6].output_pool)

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extraction(x)

            slow_feature = self.slow_avg_pool(x[0])
            fast_feature = self.fast_avg_pool(x[1])

            slow_feature = self.adp_avg_pool(slow_feature)
            fast_feature = self.adp_avg_pool(fast_feature)

        return slow_feature, fast_feature


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = slowfast()
    model = model.to(device)

    resize = args.resize

    # dicom_dir = '/mnt/gemlab_data_2/User_database/shenhao/new_metal_methods/common_data_with_new_metal_true'  # DICOM数据集的根目录
    # datainfo_llv = '/home/gem/sh/metal_simulation/Matlab_methods/final_csv_metal/new_metal_with_artifacts_with_quota_279.csv'
    
    dicom_dir = '/mnt/gemlab_data_2/User_database/shenhao/SZ_groundtruth/truth_motion_metal_data'  # DICOM数据集的根目录
    datainfo_llv = '/home/gem/sh/new_light_vqa/new_max_metal_train/some_tools/SZ_truth_data_with_motion_simulation_new_max_403_metal_with_quota.csv'
    
    
    
    transformations = transforms.Compose([transforms.Resize([resize, resize]), transforms.ToTensor(), 
                                         transforms.Normalize(mean=[0.199, 0.199, 0.199], std=[0.197, 0.197, 0.197])])

    ## 遍历根目录下的每个文件夹，寻找DICOM文件
    for root, dirs, files in os.walk(dicom_dir):
        if files:
            dicom_set = DicomDatasetExtractFeature(root, datainfo_llv, transform=transformations, resize=resize)
            dicom_loader = torch.utils.data.DataLoader(dicom_set, batch_size=1, shuffle=False, num_workers=args.num_workers)

            ## 提取特征并保存
            with torch.no_grad():
                model.eval()
                for i, (frames, series_name, dicom_name) in enumerate(dicom_loader):
                    series_name = series_name[0]
                    dicom_name = dicom_name[0]
                    save_dir = os.path.join(args.feature_save_folder, os.path.relpath(root, dicom_dir), dicom_name)
                    os.makedirs(save_dir, exist_ok=True)

                    for idx, ele in enumerate(frames):
                        ele = ele.permute(0, 2, 1, 3, 4)
                        inputs = pack_pathway_output(ele, device)
                        slow_feature, fast_feature = model(inputs)

                        if torch.isnan(slow_feature).any() or torch.isnan(fast_feature).any():
                            print("The tensor contains NaN values.")
                        
                        # 保存特征
                        np.save(os.path.join(save_dir, f'feature_{idx}_slow_feature.npy'), slow_feature.to('cpu').numpy())
                        np.save(os.path.join(save_dir, f'feature_{idx}_fast_feature.npy'), fast_feature.to('cpu').numpy())
                        if idx == 7:
                            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str, default='SlowFast')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--feature_save_folder', type=str, default='/mnt/gemlab_data_2/User_database/shenhao/SZ_groundtruth/SZ_truth_data_with_motion_simulation_need_files/temporal_feature_3_29/')

    args = parser.parse_args()

    main(args)
