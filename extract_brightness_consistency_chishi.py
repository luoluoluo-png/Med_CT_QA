import os
import math
import numpy as np
import torch
from torchvision import transforms
import pydicom
from PIL import Image
####2025.2.20，可以正常保存npy文件，并且只保存到指定路径，之前的都会乱保存
# 设置路径
imgs_path = '/mnt/gemlab_data_2/User_database/shenhao/new_metal_methods/common_data_with_new_metal_true_279file_need_files/key_frames_3_24'
save_folder = '/mnt/gemlab_data_2/User_database/shenhao/SZ_groundtruth/SZ_truth_data_with_motion_simulation_need_files/brightness_consistency_3_29'

# 图像预处理：调整大小、转换为张量并归一化
transform = transforms.Compose([
    transforms.Resize([720, 1280]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[500], std=[1000])
])

# 找出所有包含 .dcm 文件的最底层目录
dcm_dirs = []
for root, dirs, files in os.walk(imgs_path):
    if any(file.lower().endswith('.dcm') for file in files) and not dirs:
        dcm_dirs.append(root)

# 遍历包含 .dcm 文件的最底层目录
for subdir in dcm_dirs:
    print(f"Processing folder: {subdir}")

    dcm_files = [f for f in os.listdir(subdir) if f.lower().endswith('.dcm')]
    length = len(dcm_files)
    number = max(1, math.floor(length / 8))  # 确保 number 至少为 1
    print(f"Number of images to process: {number * 8}")

    brightness = torch.zeros([number * 8, 144])

    i = 0
    for dcm_file in dcm_files:
        dcm_file_path = os.path.join(subdir, dcm_file)

        # 读取 DICOM 文件
        ds = pydicom.dcmread(dcm_file_path, force=True)

        # 检查并手动设置 TransferSyntaxUID
        if not hasattr(ds.file_meta, 'TransferSyntaxUID'):
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        image_array = ds.pixel_array  # 提取像素数组

        # 归一化图像到 [0, 1]
        min_val = np.min(image_array)
        max_val = np.max(image_array)
        image_array = (image_array - min_val) / (max_val - min_val + 1e-10)  # 确保不为负

        # 转换为 PIL 图像并转换为灰度图像
        img = Image.fromarray((image_array * 255).astype(np.uint8))  # 将像素值从 [0, 1] 映射到 [0, 255]
        img = img.convert('L')
        img = transform(img)

        # 将图像重塑为 [144, 80, 80] 并展平
        img = torch.reshape(img, [144, 80, 80])
        img = torch.flatten(img, start_dim=1)

        # 计算亮度均值
        brightness_mean = torch.mean(img, dim=1)
        brightness[i] = brightness_mean
        i += 1

        if i >= number * 8:
            break

    # 打印部分数据以检查
    print(f"Brightness tensor shape: {brightness.shape}")
    print(f"Sample brightness values: {brightness[0].numpy()[:10]}")  # 打印前10个值

    # 计算亮度一致性
    brightness = torch.reshape(brightness, [8, number, 144])
    print(number, 'number')

    if number > 1:
        brightness_consistency = torch.var(brightness, dim=1, unbiased=False)
    else:
        brightness_consistency = torch.zeros([8, 144])

    # 打印亮度一致性以检查
    print(f"Brightness consistency shape: {brightness_consistency.shape}")
    print(f"Sample brightness consistency values: {brightness_consistency[0].numpy()[:10]}")  # 打印前10个值

    # 创建保存路径并保存亮度一致性结果
    relative_path = os.path.relpath(subdir, imgs_path)  # 获取相对路径
    save_path = os.path.join(save_folder, relative_path)
    os.makedirs(save_path, exist_ok=True)

    for i in range(8):
        b = brightness_consistency[i].numpy()
        b = b.reshape(1, 144, 1, 1, 1)
        file_path = os.path.join(save_path, f"brightness_consistency{i}.npy")
        if not os.path.exists(file_path):
            np.save(file_path, b)
        else:
            print(f"File {file_path} already exists, skipping.")

print("Brightness consistency analysis completed.")
