import os
import pydicom
import math
from pydicom.dataset import Dataset, FileDataset

def extract_key_frames_from_dcm(dcm_dir, save_folder):
    # 获取所有 .dcm 文件
    dcm_files = [f for f in os.listdir(dcm_dir) if f.endswith('.dcm')]
    dcm_length = len(dcm_files)
    
    if dcm_length == 0:
        print(f"No DICOM files found in {dcm_dir}")
        return
    
    # 计算需要每隔多少帧提取一个关键帧
    number = math.floor(dcm_length / 16)  # 修改为提取 16 张关键帧
    
    print(f"Processing {dcm_dir} - Total DICOM files: {dcm_length}, Extract every {number}th frame.")
    
    key_frame_idx = 0
    
    for i in range(dcm_length):
        if i % number == 0 and key_frame_idx < 16:  # 修改为 16 张关键帧
            dcm_file_path = os.path.join(dcm_dir, dcm_files[i])
            ds = pydicom.dcmread(dcm_file_path, force=True)
            
            # 创建输出目录（如果不存在）
            relative_path = os.path.relpath(dcm_dir, root_dir)  # 获取相对路径
            save_subfolder = os.path.join(save_folder, relative_path)
            exit_folder(save_subfolder)
            
            # 保存关键帧为 DICOM 文件
            save_dcm_file_path = os.path.join(save_subfolder, '{:03d}.dcm'.format(key_frame_idx))
            ds.save_as(save_dcm_file_path)
            key_frame_idx += 1

def exit_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    return

# 主函数处理根目录中的所有子目录
root_dir = '/mnt/gemlab_data_2/User_database/shenhao/new_metal_methods/common_data_with_new_metal_true'
save_folder = '/mnt/gemlab_data_2/User_database/shenhao/new_metal_methods/common_data_with_new_metal_true_279file_need_files/key_frames_3_24'  # 更新保存目录名称

for subdir, _, _ in os.walk(root_dir):  # 使用 os.walk 遍历根目录下所有子目录
    if os.path.isdir(subdir):
        extract_key_frames_from_dcm(subdir, save_folder)
