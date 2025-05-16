# import os
# import pandas as pd
# from PIL import Image

# import torch
# from torch.utils import data
# import numpy as np
# import scipy.io as scio
# import cv2
# import random
# import SimpleITK as sitk
# from skimage.transform import radon, iradon, resize
# from skimage import exposure
# import nibabel as nib
# from scipy.ndimage import rotate, shift
# from skimage.transform import resize as sk_resize
# np.random.seed(42)
# import pickle

# class VideoDataset_images(data.Dataset):
#     """Read data from the original dataset for feature extraction"""

#     def __init__(self, data_dir, filename_path, transform, database_name):
#         super(VideoDataset_images, self).__init__()


#         if database_name == 'my_train':
#             column_names = ['name', 't1','t2','mos']
#             dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
#                                    encoding="utf-8-sig")
#             self.video_names = dataInfo['name'].tolist()
#             self.score = dataInfo['mos'].tolist()

#         elif database_name == 'my_test':
#             column_names = ['name', 't1','t2','mos']
#             dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
#                                    encoding="utf-8-sig")
#             self.video_names = dataInfo['name'].tolist()
#             self.score = dataInfo['mos'].tolist()


#         self.videos_dir = data_dir
#         self.transform = transform
#         self.length = len(self.video_names)
#         self.database_name = database_name

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):

#         if self.database_name == 'my_train' or self.database_name == 'my_test':
#             video_name = self.video_names[idx]
#             video_name_str = video_name.split('.')[0]

#         video_score = torch.FloatTensor(np.array(float(self.score[idx])))
#         path_name = os.path.join(self.videos_dir, video_name_str)
#         video_channel = 3
#         video_height_crop = 720
#         video_width_crop = 1280
#         video_length_read = 8


#         key_frames = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])

#         for i in range(video_length_read):
#             imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
#             read_frame = Image.open(imge_name)
#             read_frame = read_frame.convert('RGB')
#             read_frame = self.transform(read_frame)
#             key_frames[i] = read_frame


#         return key_frames, video_score, video_name



# class VideoDataset_temporal_feature(data.Dataset):
#     def __init__(self,  temporal_feature, filename_path, database_name, feature_type):
#         super(VideoDataset_temporal_feature, self).__init__()
#         if database_name == 'my_train':
#             column_names = ['name', 't1','t2','mos']
#             dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
#                                    encoding="utf-8-sig")
#             self.video_names = dataInfo['name'].tolist()
#             self.score = dataInfo['mos'].tolist()

#         elif database_name == 'my_test':
#             column_names = ['name', 't1','t2','mos']
#             dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
#                                    encoding="utf-8-sig")
#             self.video_names = dataInfo['name'].tolist()
#             self.score = dataInfo['mos'].tolist()

#         self.temporal_feature = temporal_feature
#         self.length = len(self.video_names)
#         self.feature_type = feature_type
#         self.database_name = database_name

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         if self.database_name == 'my_train' or self.database_name == 'my_test':
#             video_name = self.video_names[idx]
#             video_name_str = video_name.split('.')[0]

#         video_score = torch.FloatTensor(np.array(float(self.score[idx])))
#         video_length_read = 8

#         # read temporal features

#         if self.feature_type == 'SlowFast':
#             feature_folder_name = os.path.join(self.temporal_feature, video_name_str)
#             temporal_feature = torch.zeros([video_length_read, 2048 + 256])
#             for i in range(video_length_read):
#                 i_index = i
#                 feature_3D_slow = np.load(
#                     os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
#                 feature_3D_slow = torch.from_numpy(feature_3D_slow)
#                 feature_3D_slow = feature_3D_slow.squeeze()
#                 feature_3D_fast = np.load(
#                     os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
#                 feature_3D_fast = torch.from_numpy(feature_3D_fast)
#                 feature_3D_fast = feature_3D_fast.squeeze()
#                 feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
#                 temporal_feature[i] = feature_3D

#         return  temporal_feature, video_score, video_name


# class VideoDataset_extract_temporal_feature(data.Dataset):
#     """Read data from the original dataset for feature extraction"""

#     def __init__(self, data_dir, filename_path, transform, resize):
#         super(VideoDataset_extract_temporal_feature, self).__init__()
#         column_names = ['name', 't1', 't2', 'mos']

#         dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
#                                encoding="utf-8-sig")

#         self.video_names = dataInfo['name']
#         self.score = dataInfo['mos']
#         self.videos_dir = data_dir
#         self.transform = transform
#         self.resize = resize
#         self.length = len(self.video_names)

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         video_name = self.video_names.iloc[idx]
#         video_score = torch.FloatTensor(np.array(float(self.score.iloc[idx]))) / 20

#         filename = os.path.join(self.videos_dir, video_name)

#         video_capture = cv2.VideoCapture()
#         video_capture.open(filename)
#         cap = cv2.VideoCapture(filename)

#         video_channel = 3

#         video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

#         if video_frame_rate == 0:
#             video_clip = 10
#         else:
#             video_clip = int(video_length / video_frame_rate)



#         video_clip_min = 8

#         video_length_clip = 32

#         transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

#         transformed_video_all = []

#         video_read_index = 0
#         for i in range(video_length):
#             has_frames, frame = video_capture.read()
#             if has_frames:
#                 read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#                 read_frame = self.transform(read_frame)
#                 transformed_frame_all[video_read_index] = read_frame
#                 video_read_index += 1

#         if video_read_index < video_length:
#             for i in range(video_read_index, video_length):
#                 transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]

#         video_capture.release()

#         for i in range(video_clip):
#             transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
#             if (i * video_frame_rate + video_length_clip) <= video_length:
#                 transformed_video = transformed_frame_all[
#                                     i * video_frame_rate: (i * video_frame_rate + video_length_clip)]
#             else:
#                 transformed_video[:(video_length - i * video_frame_rate)] = transformed_frame_all[i * video_frame_rate:]
#                 for j in range((video_length - i * video_frame_rate), video_length_clip):
#                     transformed_video[j] = transformed_video[video_length - i * video_frame_rate - 1]
#             transformed_video_all.append(transformed_video)

#         if video_clip < video_clip_min:
#             for i in range(video_clip, video_clip_min):
#                 transformed_video_all.append(transformed_video_all[video_clip - 1])

#         return transformed_video_all, video_score, video_name


# import pydicom
# class DicomDatasetExtractFeature(data.Dataset):
#     """Read data from DICOM files for feature extraction"""

#     def __init__(self, data_dir, filename_path, transform, resize):
#         super(DicomDatasetExtractFeature, self).__init__()
#         column_names = ['name', 'mos']

#         data_info = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")

#         self.dicom_dirs = data_info['name']  # Folder names containing DICOM files
#         self.scores = data_info['mos']
#         self.data_dir = data_dir
#         self.transform = transform
#         self.resize = resize
#         self.length = len(self.dicom_dirs)

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         dicom_dir = self.dicom_dirs.iloc[idx]
#         score = torch.FloatTensor(np.array(float(self.scores.iloc[idx]))) / 20

#         dicom_folder_path = os.path.join(self.data_dir, dicom_dir[1:-1])
#         dicom_files = sorted(os.listdir(dicom_folder_path))
        
#         channel = 3 # Assuming grayscale images
#         dicom_length = len(dicom_files)

#         transformed_frames = torch.zeros([dicom_length, channel, self.resize, self.resize])

#         # Read and transform each DICOM file
#         for i, dicom_file in enumerate(dicom_files):
#             dicom_path = os.path.join(dicom_folder_path, dicom_file)
#             if os.path.isfile(dicom_path) and dicom_file.lower().endswith('.dcm'):
                
#                 dicom_data = pydicom.dcmread(dicom_path,force=True)
#                 image = dicom_data.pixel_array

#                 if len(image.shape) == 2:  # 单通道灰度图像
#                     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # 转换为三通道灰度图像 (BGR)
#                 # 使用 cv2.cvtColor 转换为 RGB 格式
#                 image = (image / np.max(image) * 255).astype(np.uint8)

#                 image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
#                 # 转换为 PIL 图像
#                 image = Image.fromarray(image_rgb)

#                 image = self.transform(image)
#                 transformed_frames[i] = image

#         video_clip_min = 8
#         video_length_clip = 32

#         transformed_videos = []

#         for i in range(dicom_length // video_length_clip):
#             start_idx = i * video_length_clip
#             end_idx = start_idx + video_length_clip
#             transformed_videos.append(transformed_frames[start_idx:end_idx])

#         # Ensure at least 8 video clips
#         if len(transformed_videos) < video_clip_min:
#             last_clip = transformed_videos[-1]
#             for _ in range(len(transformed_videos), video_clip_min):
#                 transformed_videos.append(last_clip)

#         return transformed_videos, score, dicom_dir[1:-1]


# class VideoDataset_images_with_temporal_features(data.Dataset):
#     """Read data from the original dataset for feature extraction"""

#     def __init__(self, data_dir, data_dir_3D, filename_path, transform, database_name, feature_type):
#         super(VideoDataset_images_with_temporal_features, self).__init__()


#         if database_name == 'my_train':
#             column_names = ['name', 't1','t2','mos']
#             dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
#                                    encoding="utf-8-sig")
#             self.video_names = dataInfo['name'].tolist()
#             self.score = dataInfo['mos'].tolist()

#         elif database_name == 'my_test':
#             column_names = ['name', 't1','t2','mos']
#             dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
#                                    encoding="utf-8-sig")
#             self.video_names = dataInfo['name'].tolist()
#             self.score = dataInfo['mos'].tolist()


#         self.videos_dir = data_dir
#         self.data_dir_3D = data_dir_3D
#         self.transform = transform
#         self.length = len(self.video_names)
#         self.feature_type = feature_type
#         self.database_name = database_name

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):

#         if self.database_name == 'my_train' or self.database_name == 'my_test':
#             video_name = self.video_names[idx]
#             video_name_str = video_name.split('.')[0]

#         video_score = torch.FloatTensor(np.array(float(self.score[idx])))

#         path_name = os.path.join(self.videos_dir, video_name_str)

#         video_channel = 3
#         video_height_crop = 720
#         video_width_crop = 1280
#         video_length_read = 8


#         transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])

#         for i in range(video_length_read):
#             imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
#             read_frame = Image.open(imge_name)
#             read_frame = read_frame.convert('RGB')
#             read_frame = self.transform(read_frame)
#             transformed_video[i] = read_frame

#         # read temporal features

#         if self.feature_type == 'SlowFast':
#             feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
#             transformed_feature = torch.zeros([video_length_read, 2048 + 256])
#             for i in range(video_length_read):
#                 i_index = i
#                 feature_3D_slow = np.load(
#                     os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
#                 feature_3D_slow = torch.from_numpy(feature_3D_slow)
#                 feature_3D_slow = feature_3D_slow.squeeze()
#                 feature_3D_fast = np.load(
#                     os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
#                 feature_3D_fast = torch.from_numpy(feature_3D_fast)
#                 feature_3D_fast = feature_3D_fast.squeeze()
#                 feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
#                 transformed_feature[i] = feature_3D

#         return transformed_video, transformed_feature, video_score, video_name


# class VideoDataset_spatio_temporal_brightness(data.Dataset):
#     """Read data from the original dataset for feature extraction"""

#     def __init__(self, data_dir, data_dir_3D, brightness, filename_path, transform, database_name, feature_type):
#         super(VideoDataset_spatio_temporal_brightness, self).__init__()


#         if database_name == 'my_train':
#             column_names = ['name', 't1','t2','mos']
#             dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
#                                    encoding="utf-8-sig")
#             self.video_names = dataInfo['name'].tolist()
#             self.score = dataInfo['mos'].tolist()

#         elif database_name == 'my_test':
#             column_names = ['name', 't1','t2','mos']
#             dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
#                                    encoding="utf-8-sig")
#             self.video_names = dataInfo['name'].tolist()
#             self.score = dataInfo['mos'].tolist()


#         self.videos_dir = data_dir
#         self.data_dir_3D = data_dir_3D
#         self.transform = transform
#         self.length = len(self.video_names)
#         self.feature_type = feature_type
#         self.database_name = database_name
#         self.brightness = brightness

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):

#         if self.database_name == 'my_train' or self.database_name == 'my_test':
#             video_name = self.video_names[idx]
#             video_name_str = video_name.split('.')[0]

#         video_score = torch.FloatTensor(np.array(float(self.score[idx])))

#         path_name = os.path.join(self.videos_dir, video_name_str)

#         video_channel = 3
#         video_height_crop = 512
#         video_width_crop = 512
#         video_length_read = 8


#         transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])
#         j = 0
#         for img in os.listdir(path_name):
#             imge_name = os.path.join(path_name, img)
#             read_frame = Image.open(imge_name)
#             read_frame = read_frame.convert('RGB')
#             read_frame = self.transform(read_frame)
#             transformed_video[j] = read_frame
#             j += 1

#         # read temporal features

#         if self.feature_type == 'SlowFast':
#             feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
#             brightness_folder_name = os.path.join(self.brightness, video_name_str)
#             transformed_feature = torch.zeros([video_length_read, 2048 + 256 + 144])
#             for i in range(video_length_read):
#                 i_index = i

#                 feature_3D_slow = np.load(
#                     os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
#                 feature_3D_slow = torch.from_numpy(feature_3D_slow)
#                 feature_3D_slow = feature_3D_slow.squeeze()

#                 feature_3D_fast = np.load(
#                     os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
#                 feature_3D_fast = torch.from_numpy(feature_3D_fast)
#                 feature_3D_fast = feature_3D_fast.squeeze()

#                 brightness_consistency = np.load(
#                     os.path.join(brightness_folder_name, 'brightness_consistency' + str(i_index) + '.npy')
#                 )
#                 brightness_consistency = torch.from_numpy(brightness_consistency)
#                 brightness_consistency = brightness_consistency.squeeze()
#                 brightness_consistency *= 10

#                 feature_3D = torch.cat([feature_3D_slow, feature_3D_fast, brightness_consistency])
#                 transformed_feature[i] = feature_3D

#         return transformed_video, transformed_feature, video_score, video_name
    

# class DicomDataset_spatio_temporal_brightness(data.Dataset):
#     """Read DICOM data for feature extraction"""

#     def __init__(self, split,dcm_path,data_dir, data_dir_3D, brightness, filename_path, transform, database_name, feature_type,
#                  rotation_angle=(0, 10), translation=(-30, 30)):
#         super(DicomDataset_spatio_temporal_brightness, self).__init__()
#         self.video_names = []  # 初始化视频名称列表
#         if database_name == 'my_train' or database_name == 'my_test':
#             column_names = ['name', 'mos']
#             dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
#                                    encoding="utf-8-sig")
#             self.video_names = dataInfo['name'].tolist()
#             self.score = dataInfo['mos'].tolist()
#         self.split = split 
#         self.dcm_path = dcm_path
#         self.videos_dir = data_dir
#         self.data_dir_3D = data_dir_3D
#         self.transform = transform
#         self.length = len(self.video_names)
#         self.feature_type = feature_type
#         self.database_name = database_name
#         self.brightness = brightness

#         self.rotation_angle = rotation_angle
#         self.translation = translation
#         self.rotation_angle = (-70, 70)
#         self.translation = (50, 150)

#         self.augmentation_probability = 0.6  # 可以根据需求调整
#         self.motion_artifact_probability=0.5
#         self.metal_artifact_probability=0.3
#         # with open("/home/data/shenhao/lung_quality/med/seg/"+split+"_data.txt", "rb") as fp:     #使用 with 语句打开文件 "split_data.txt",split可以是train，val,test,以二进制读取模式打开文件
#         #     self.file_list = pickle.load(fp)            #从打开的fp文件中加载数据，并将加载的数据赋值给类的实例变量 self.file_list。
#         #     #print(self.file_list,"list")                #打印 self.file_list 的内容。这行代码用于检查加载的数据，以确保正确读取了文件。
#         # with open("/home/data/shenhao/lung_quality/med/seg/"+split+"_labels.txt", "rb") as fpl:
#         #     self.labels = pickle.load(fpl)
#         #     #print(self.labels,"labels")
#         # self.size = len(self.file_list)                                                      #计算加载的数据集的大小，即 self.file_list 中的元素个数，
        

#     def __len__(self):
        
#         return self.length

#     def __getitem__(self, idx):
#         # fname = self.file_list[index]           #通过索引 index 获取 self.file_list 中对应位置的文件名， "xyz.nii"
#         # fpath = os.path.join(self.data_dir, str(fname))
#         if self.database_name == 'my_train' or self.database_name == 'my_test':
#             video_name = self.video_names[idx]
#             video_name_str = str(video_name)[1:-1]

#             #print(f"video_name: {video_name}, type: {type(video_name)}")
#             # video_name_str = video_name.split('.')[0]
#             #video_name_str =  video_name[1:-1]
            

#         video_score = torch.FloatTensor(np.array(float(self.score[idx])))

#         path_name = os.path.join(self.videos_dir, video_name_str)
#         dcm_path = os.path.join(self.dcm_path, video_name_str)
#         # print(video_name_str,'这是video_name')  ##video_name_str就是对于的文件名
#         # print(dcm_path,'这是dcmpath')

#         rotation_angle = 0
#         three_D_rotation_angle=[0,0,0]
#         translation = [0.0, 0.0, 0.0]
#         motion_translation = [0.0, 0.0, 0.0]
#         label3=0
        
        
#         def load_metal_artifact_labels(file_path):
#             # 读取 CSV 文件
#             data = pd.read_csv(file_path, header=0)  # header=0 指定第一行作为列名
#             # 去掉 'name' 列中的引号和多余的空格，确保匹配时的一致性
#             # 创建字典，键为 name，值为 mos
#             return dict(zip(data['name'],data['metal']))
#         def load_motion_artifact_labels(file_path):
#             # 读取 CSV 文件
#             data = pd.read_csv(file_path, header=0)  # header=0 指定第一行作为列名
#             # 去掉 'name' 列中的引号和多余的空格，确保匹配时的一致性
            
#             # 创建字典，键为 name，值为 mos
#             return dict(zip(data['name'],data['motion']))
#         DATAINFO_TEST = '/home/gem/sh/new_light_vqa/csv_label/SZ_true_619data_with_true_label_3_31.csv'
#         metal_artifact_labels = load_metal_artifact_labels(DATAINFO_TEST)
#         motion_artifact_labels = load_motion_artifact_labels(DATAINFO_TEST)
        
#         # print(metal_artifact_labels,'****这是metal_artifact_labels****')
#         # print(motion_artifact_labels,'****这是motion_artifact_labels****')
#         #folder_name = video_name_str.item()  # 获取文件夹名
#         video_name_str_cleaned = video_name_str.strip("'")
#         video_name_str_cleaned = f"'{video_name_str}'"
#         #print(video_name_str_cleaned,'****这是video_name_str_cleaned')
#         label4=0
#         # 查找标签
#         label4 = metal_artifact_labels.get(video_name_str_cleaned)  # 直接读取标签
#         #label4 = metal_artifact_labels.get(video_name_str,0)  # 直接读取标签
#         #print(label4,video_name_str,'****这是label4的值和对应的文件名****')
#         label3 = motion_artifact_labels.get(video_name_str_cleaned)
#         #print(label3,video_name_str,'****这是label3的值和对应的文件名****')
        
#         #image_array = np.zeros((1, 512, 512), dtype=np.float32)  # 默认形状为 (1, 512, 512)
#         image_array, image3D = read_dicom_series_as_numpy_array_single(dcm_path)
#         max_value = np.max(image_array)
#         print(max_value,'这是第一个 image_array max_value')
        
#         if check_image_size(dcm_path):
#             image_array, image3D = read_dicom_series_as_numpy_array(dcm_path)
#             max_value = np.max(image_array)
#             print(max_value,'这是check_image_size里的 image_array max_value')
#         else:
#             print("DICOM 文件尺寸不符合要求，跳过")
        
#         print(image_array.shape,'shape,image array')
#         #image_array = np.array([resize(image, (512, 512), order=2) for image in image_array])
#         # 1. 选择其中的 3 个层作为通道
#         # 假设我们选择前 3 个层来作为通道


        

#         if random.random() < self.augmentation_probability:
#             rotation_angle = np.random.randint(self.rotation_angle[0], self.rotation_angle[1])
#             three_D_rotation_angle=[0,0,rotation_angle]
#             translation = [
#                 np.random.randint(self.translation[0], self.translation[1]),
#                 np.random.randint(self.translation[0], self.translation[1]),
#                 0
#             ]
#             #print(f"进行了旋转和平移，旋转角度: {rotation_angle}, 平移参数: {translation}")
            
#             image_array, transformed_image = rotate_and_translate(image3D, three_D_rotation_angle, translation)
            
#             max_value = np.max(image_array)
#             print(max_value,'这是 旋转后的 image_array max_value')
            
#               # 直接读取标签
#             # if random.random() < self.motion_artifact_probability and label3 == 0:
#             #     random_y_translation = np.random.randint(0, 11)  # 在0到10之间生成一个随机整数作为y分量
#             #     random_x_translation = np.random.randint(0, 11)
#             #     random_z_translation = np.random.randint(0, 11)  
#             #     motion_translation = [random_x_translation, random_y_translation,random_z_translation]  # x分量为0，y分量为随机数
#             #     image_array = simulate_motion_blur(image_array, motion_translation)
#             #     point = np.array([random_x_translation, random_y_translation,random_z_translation])
#             #         # 计算点到原点的欧氏距离
#             #     distance_to_origin = np.linalg.norm(point)
#             #     #label3=distance_to_origin/10
#             #     label3=1
            
            
#         # if random.random() < self.metal_artifact_probability and label4 == 0:
#         #     image_array = simulate_metal_artifact(image_array)
#         #     label4=1

#         #nii_path = os.path.join(dcm_path, f"{str(video_name_str).split('.')[0]}.nii")
        
#         ##################
#         # extracted_value = dcm_path.split('/')[-1]
       
#         # segment_path='/mnt/gemlab_data_2/User_database/shenhao/whole_sz_and_luna16/whole_sz_and_luna16_data'##用原来的分割先试一下
#         # #nii_path = os.path.join(dcm_path, f"{str(video_name_str).split('.')[0]}_segment.nii")
#         # segment_path = os.path.join(segment_path, f"{str(video_name_str)}")  ###2025.2.20自己加的
        
#         # nii_path = os.path.join(segment_path, f"{str(extracted_value)}_segment.nii")
#         ##################
#         extracted_value = dcm_path.split('/')[-1]
       
#         segment_path='/mnt/gemlab_data_2/User_database/shenhao/whole_sz_and_luna16/whole_sz_and_luna16_data'##用原来的分割先试一下
#         #nii_path = os.path.join(dcm_path, f"{str(video_name_str).split('.')[0]}_segment.nii")
#         segment_path = os.path.join(segment_path, f"{str(video_name_str)}")  ###2025.2.20自己加的
        
#         nii_path = os.path.join(segment_path, f"{str(extracted_value)}_segment.nii")
#         #nii_path = os.path.join(dcm_path, f"{str(video_name_str).split('.')[0]}_segment.nii")
#         #nii_path = os.path.join(dcm_path, f"{str(video_name_str).split('.')[0]}_segment.nii")

#         #print(nii_path,'这是nii_path')
#         #nii_path=os.path.isfile(dicom_path) and dicom_file.lower().endswith('.nii')
#         voxel_spacing = get_voxel_spacing(nii_path)
#         voxel_size = np.array(voxel_spacing)
#         nii_data, affine = load_nii_file(nii_path)
#         original_volume = calculate_original_volume(nii_data, voxel_size)

#         transformed_volume, transformed_mask = calculate_transformed_volume(nii_data, voxel_size, rotation_angle=rotation_angle,translation=translation)
#         volume_percentage = calculate_volume_percentage(original_volume, transformed_volume)
        
#         print(f"这是ground_train训练的sh3-原始体积:{original_volume / 1000},分割后的体积: {transformed_volume / 1000}，比率: {volume_percentage},文件名:{video_name_str},旋转参数:{rotation_angle}平移参数:{translation}")
#         epsilon = 1e-10
        
#         max_value = np.max(image_array)
#         print(max_value,'这是取3个之前的 image_array max_value')
        
#         max_values_per_layer = np.max(image_array, axis=(1, 2))  # 沿着 512x512 的维度计算每层的最大值

#         # 获取最大值最大的 3 个索引
#         top_3_indices = np.argsort(max_values_per_layer)[-3:]  # 获取最大值的索引，按值排序并选择最后 3 个最大值

#         # 根据这些索引选择相应的层
#         image_array_selected = image_array[top_3_indices, :, :]
#         image_array_selected = np.array([resize(image, (512, 512), order=2, preserve_range=True) for image in image_array_selected])
#         max_value = np.max(image_array_selected)
#         print(max_value,'这是取3个之后的image_array_selected 的 max_value')
#         print(image_array_selected.shape,'image_array_selected de shape')
#         if max_value>=2900:
#             label4=1
#         else:
#             label4=0
        
#         # 2. 转换为 PyTorch 张量
#         image_tensor = torch.from_numpy(image_array_selected)  # 转为张量

#         # 3. 将其变成 [1, 3, 512, 512]，即添加批次维度
#         #image_tensor = image_tensor.unsqueeze(0)  # 扩展批次维度，形状变为 [1, 3, 512, 512]
#         image_tensor = image_tensor.to(torch.float32)

#         # 4. 查看结果
#         print(image_tensor.shape,'image_tensor de shape')  # 应该是 [1, 3, 512, 512]
#         #max_value = np.max(image_array)
#         # print(max_value,'这是 max_value')
#         # print(image_array.shape,'image_array de shape')   #1890 这是 max_value,(70, 512, 512) image_array de shape
#         image_array = np.array([resize(image, (512, 512), order=2) for image in image_array])
#         image_array = (image_array-image_array.min())  / (image_array.max()  - image_array.min() + epsilon)         #归一化通过这个操作，图像的所有像素值都被映射到 [0, 1] 的范围内
#         image_array = (image_array * 255).astype(np.uint8)
        
#         if rotation_angle>0:
#             rotation_angle=rotation_angle/360
#         if rotation_angle<0:
#             rotation_angle=rotation_angle/(-360)
#         if rotation_angle==0:
#             rotation_angle=0
#         #label = int(self.labels[index])
#         label1 = rotation_angle
#         label2 = volume_percentage
        
#         label1 = torch.FloatTensor(np.array(float(label1)))
#         label2 = torch.FloatTensor(np.array(float(label2)))

#         video_channel = 3
#         video_height_crop = 512
#         video_width_crop = 512
#         video_length_read = 1    #####改完batch的话要来这里改，如果batch=1，则这里1，

#         transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])
#         j = 0
#         # max_value = np.max(image_array)
#         # print(max_value,'这是 max_value')    #254 这是 max_value
#         for image_i in range(len(image_array)):
#             image = image_array[image_i]

#             # 如果是灰度图像，将其转换为三通道图像
#             if len(image.shape) == 2:  # 单通道灰度图像
#                 image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # 转换为三通道图像 (BGR)

#             # 转换为 uint8 类型
#             image = (image / np.max(image) * 255).astype(np.uint8)

#             # 转换为 RGB 格式
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             # 转换为 PIL 图像并应用 transform
#             read_frame = Image.fromarray(image_rgb)
#             read_frame = self.transform(read_frame)
#             transformed_video[j] = read_frame
#             j += 1
#             if j >= video_length_read:
#                 break

#         # 读取时空亮度一致性和 3D 特征
#         if self.feature_type == 'SlowFast':
#             feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
#             brightness_folder_name = os.path.join(self.brightness, video_name_str)
#             transformed_feature = torch.zeros([video_length_read, 2048 + 256 + 144])

#             for i in range(video_length_read):
#                 i_index = i

#                 feature_3D_slow = np.load(
#                     os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
#                 feature_3D_slow = torch.from_numpy(feature_3D_slow).squeeze()

#                 feature_3D_fast = np.load(
#                     os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
#                 feature_3D_fast = torch.from_numpy(feature_3D_fast).squeeze()

#                 brightness_consistency = np.load(
#                     os.path.join(brightness_folder_name, 'brightness_consistency' + str(i_index) + '.npy')
#                 )
#                 brightness_consistency = torch.from_numpy(brightness_consistency).squeeze()
#                 brightness_consistency *= 10

#                 feature_3D = torch.cat([feature_3D_slow, feature_3D_fast, brightness_consistency])
#                 transformed_feature[i] = feature_3D
                

#                 print(transformed_video.shape,'transformed_video de shape')
#         return image_tensor, transformed_feature, video_name, label1,label2,label3,label4
    

# def read_dicom_series_as_numpy_array_single(file_path):
#     series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(file_path)
#     if not series_IDs:
#         raise ValueError(f"在 {file_path} 中找不到 DICOM 系列")
    
#     series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file_path, series_IDs[0])
#     series_reader = sitk.ImageSeriesReader()
#     series_reader.SetFileNames(series_file_names)
#     image3D = series_reader.Execute()
#     image_array = sitk.GetArrayFromImage(image3D)
    
#     return image_array, image3D
# import os
# import SimpleITK as sitk
# import numpy as np
# def check_image_size(dcm_path):
#     """
#     检查 DICOM 文件夹中的所有 DICOM 文件的尺寸是否为 (512, 512)
#     如果不是 512x512，返回 False，表示跳过。
#     """
#     dicom_files = [f for f in os.listdir(dcm_path) if f.endswith('.dcm')]
    
#     if not dicom_files:
#         raise ValueError(f"在 {dcm_path} 中找不到 DICOM 文件 (.dcm)")
    
#     # 读取第一个 DICOM 文件来检查尺寸
#     first_dcm_file = os.path.join(dcm_path, dicom_files[0])
#     image = sitk.ReadImage(first_dcm_file)
    
#     size = image.GetSize()  # 获取图像的尺寸 (宽度, 高度, 深度)
    
    
#     # 检查是否符合 (512, 512)
#     if size[0] != 512 or size[1] != 512:
#         print(f"跳过 {dcm_path}，因为尺寸不是 512x512")
#         return False  # 如果尺寸不是 512x512，返回 False
    
#     return True  # 如果尺寸符合要求，返回 True
# def read_dicom_series_as_numpy_array(file_path):
#     # 获取路径下所有的 .dcm 文件
#     dicom_files = [f for f in os.listdir(file_path) if f.endswith('.dcm')]
#     if not dicom_files:
#         raise ValueError(f"在 {file_path} 中找不到 DICOM 文件 (.dcm)")

#     valid_dicom_files = []
#     for dicom_file in dicom_files:
#         dicom_file_path = os.path.join(file_path, dicom_file)
#         # 读取单个 DICOM 文件
#         image = sitk.ReadImage(dicom_file_path)
#         size = image.GetSize()
#         if size[0] == 512 and size[1] == 512:
#             valid_dicom_files.append(dicom_file_path)

#     if not valid_dicom_files:
#         raise ValueError(f"在 {file_path} 中未找到尺寸为 512x512 的 DICOM 文件")

#     # 读取 DICOM 文件序列
#     series_reader = sitk.ImageSeriesReader()
#     series_reader.SetFileNames(valid_dicom_files)
#     # 读取图像数据
#     image3D = series_reader.Execute()
#     # 将 SimpleITK 图像转换为 NumPy 数组
#     image_array = sitk.GetArrayFromImage(image3D)
#     return image_array, image3D



# def find_lung_center(image3D):
#     # 假设 image3D 是一个包含多个通道的图像，使用 Cast 转换为单通道图像
#     #image3D = sitk.Cast(image3D, sitk.sitkInt16)  # 你可以根据需求选择合适的像素类型
#     # 第一步，将图像转换为 uint16
#     image3D = sitk.Cast(image3D, sitk.sitkUInt16)

#     # 第二步，将图像转换为 float32
#     image3D = sitk.Cast(image3D, sitk.sitkFloat32)


#     lung_mask = sitk.BinaryThreshold(image3D, lowerThreshold=-1024, upperThreshold=0, insideValue=1, outsideValue=0)
#     label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
#     label_shape_filter.Execute(lung_mask)
#     try:
#         center_of_mass = label_shape_filter.GetCentroid(1)
#     except RuntimeError:
#         print("Label 1 not found. Using default center or skipping.")
#         center_of_mass = [0, 0, 0]  # 或者提供默认值

#     #center_of_mass = label_shape_filter.GetCentroid(1) 原来的，我改成try
    
#     return center_of_mass

# def rotate_and_translate(image3D, rotation_angles_deg, translation_vector):
#     rotation_angles_rad = np.deg2rad(rotation_angles_deg)
#     center_of_mass = find_lung_center(image3D)
#     rotation_transform = sitk.Euler3DTransform()
#     rotation_transform.SetCenter(center_of_mass)
#     rotation_transform.SetRotation(rotation_angles_rad[0], rotation_angles_rad[1], rotation_angles_rad[2])
#     rotated_image = sitk.Resample(image3D, image3D, rotation_transform, sitk.sitkLinear, -1024.0)
#     translation_transform = sitk.TranslationTransform(3)
#     translation_transform.SetOffset(translation_vector)
#     transformed_image = sitk.Resample(rotated_image, image3D, translation_transform, sitk.sitkLinear, -1024.0)
#     transformed_image_np = sitk.GetArrayFromImage(transformed_image)
    
#     return transformed_image_np, transformed_image


# def get_voxel_spacing(file_path):
#     nii_img = nib.load(file_path)
#     header = nii_img.header
#     voxel_spacing = header.get_zooms()
#     return voxel_spacing

# def load_nii_file(file_path):
#     nii_img = nib.load(file_path)
#     nii_data = nii_img.get_fdata()
#     return nii_data, nii_img.affine

# def calculate_original_volume(mask, voxel_size):
#     return np.sum(mask) * np.prod(voxel_size)

# def calculate_transformed_volume(mask, voxel_size, rotation_angle, translation):
#     # 进行旋转
#     rotated_mask = rotate(mask, angle=rotation_angle, reshape=False, mode='constant', cval=mask.min())

#     # 进行平移
#     translation_matrix = np.array([translation[0], translation[1],translation[2]])
#     translated_mask = shift(rotated_mask, translation_matrix)

#     # 计算旋转平移后的体积
#     transformed_volume = np.sum(translated_mask) * np.prod(voxel_size)
#     return transformed_volume, translated_mask

# def calculate_volume_percentage(original_volume, translated_volume):
#     return (translated_volume / original_volume)







import os
import pandas as pd
from PIL import Image

import torch
from torch.utils import data
import numpy as np
import scipy.io as scio
import cv2
import random
import SimpleITK as sitk
from skimage.transform import radon, iradon, resize
from skimage import exposure
import nibabel as nib
from scipy.ndimage import rotate, shift
from skimage.transform import resize as sk_resize
np.random.seed(42)
import pickle

class VideoDataset_images(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, filename_path, transform, database_name):
        super(VideoDataset_images, self).__init__()


        if database_name == 'my_train':
            column_names = ['name', 't1','t2','mos']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'my_test':
            column_names = ['name', 't1','t2','mos']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()


        self.videos_dir = data_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if self.database_name == 'my_train' or self.database_name == 'my_test':
            video_name = self.video_names[idx]
            video_name_str = video_name.split('.')[0]

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))
        path_name = os.path.join(self.videos_dir, video_name_str)
        video_channel = 3
        video_height_crop = 720
        video_width_crop = 1280
        video_length_read = 8


        key_frames = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])

        for i in range(video_length_read):
            imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            key_frames[i] = read_frame


        return key_frames, video_score, video_name



class VideoDataset_temporal_feature(data.Dataset):
    def __init__(self,  temporal_feature, filename_path, database_name, feature_type):
        super(VideoDataset_temporal_feature, self).__init__()
        if database_name == 'my_train':
            column_names = ['name', 't1','t2','mos']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'my_test':
            column_names = ['name', 't1','t2','mos']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        self.temporal_feature = temporal_feature
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name == 'my_train' or self.database_name == 'my_test':
            video_name = self.video_names[idx]
            video_name_str = video_name.split('.')[0]

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))
        video_length_read = 8

        # read temporal features

        if self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.temporal_feature, video_name_str)
            temporal_feature = torch.zeros([video_length_read, 2048 + 256])
            for i in range(video_length_read):
                i_index = i
                feature_3D_slow = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                temporal_feature[i] = feature_3D

        return  temporal_feature, video_score, video_name


class VideoDataset_extract_temporal_feature(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, filename_path, transform, resize):
        super(VideoDataset_extract_temporal_feature, self).__init__()
        column_names = ['name', 't1', 't2', 'mos']

        dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                               encoding="utf-8-sig")

        self.video_names = dataInfo['name']
        self.score = dataInfo['mos']
        self.videos_dir = data_dir
        self.transform = transform
        self.resize = resize
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names.iloc[idx]
        video_score = torch.FloatTensor(np.array(float(self.score.iloc[idx]))) / 20

        filename = os.path.join(self.videos_dir, video_name)

        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap = cv2.VideoCapture(filename)

        video_channel = 3

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

        if video_frame_rate == 0:
            video_clip = 10
        else:
            video_clip = int(video_length / video_frame_rate)



        video_clip_min = 8

        video_length_clip = 32

        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

        transformed_video_all = []

        video_read_index = 0
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                read_frame = self.transform(read_frame)
                transformed_frame_all[video_read_index] = read_frame
                video_read_index += 1

        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]

        video_capture.release()

        for i in range(video_clip):
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            if (i * video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[
                                    i * video_frame_rate: (i * video_frame_rate + video_length_clip)]
            else:
                transformed_video[:(video_length - i * video_frame_rate)] = transformed_frame_all[i * video_frame_rate:]
                for j in range((video_length - i * video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i * video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])

        return transformed_video_all, video_score, video_name


import pydicom
class DicomDatasetExtractFeature(data.Dataset):
    """Read data from DICOM files for feature extraction"""

    def __init__(self, data_dir, filename_path, transform, resize):
        super(DicomDatasetExtractFeature, self).__init__()
        column_names = ['name', 'mos']

        data_info = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")

        self.dicom_dirs = data_info['name']  # Folder names containing DICOM files
        self.scores = data_info['mos']
        self.data_dir = data_dir
        self.transform = transform
        self.resize = resize
        self.length = len(self.dicom_dirs)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dicom_dir = self.dicom_dirs.iloc[idx]
        score = torch.FloatTensor(np.array(float(self.scores.iloc[idx]))) / 20

        dicom_folder_path = os.path.join(self.data_dir, dicom_dir[1:-1])
        dicom_files = sorted(os.listdir(dicom_folder_path))
        
        channel = 3 # Assuming grayscale images
        dicom_length = len(dicom_files)

        transformed_frames = torch.zeros([dicom_length, channel, self.resize, self.resize])

        # Read and transform each DICOM file
        for i, dicom_file in enumerate(dicom_files):
            dicom_path = os.path.join(dicom_folder_path, dicom_file)
            if os.path.isfile(dicom_path) and dicom_file.lower().endswith('.dcm'):
                
                dicom_data = pydicom.dcmread(dicom_path,force=True)
                image = dicom_data.pixel_array

                if len(image.shape) == 2:  # 单通道灰度图像
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # 转换为三通道灰度图像 (BGR)
                # 使用 cv2.cvtColor 转换为 RGB 格式
                image = (image / np.max(image) * 255).astype(np.uint8)

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 转换为 PIL 图像
                image = Image.fromarray(image_rgb)

                image = self.transform(image)
                transformed_frames[i] = image

        video_clip_min = 8
        video_length_clip = 32

        transformed_videos = []

        for i in range(dicom_length // video_length_clip):
            start_idx = i * video_length_clip
            end_idx = start_idx + video_length_clip
            transformed_videos.append(transformed_frames[start_idx:end_idx])

        # Ensure at least 8 video clips
        if len(transformed_videos) < video_clip_min:
            last_clip = transformed_videos[-1]
            for _ in range(len(transformed_videos), video_clip_min):
                transformed_videos.append(last_clip)

        return transformed_videos, score, dicom_dir[1:-1]


class VideoDataset_images_with_temporal_features(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, data_dir_3D, filename_path, transform, database_name, feature_type):
        super(VideoDataset_images_with_temporal_features, self).__init__()


        if database_name == 'my_train':
            column_names = ['name', 't1','t2','mos']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'my_test':
            column_names = ['name', 't1','t2','mos']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()


        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if self.database_name == 'my_train' or self.database_name == 'my_test':
            video_name = self.video_names[idx]
            video_name_str = video_name.split('.')[0]

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)

        video_channel = 3
        video_height_crop = 720
        video_width_crop = 1280
        video_length_read = 8


        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])

        for i in range(video_length_read):
            imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame

        # read temporal features

        if self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048 + 256])
            for i in range(video_length_read):
                i_index = i
                feature_3D_slow = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D

        return transformed_video, transformed_feature, video_score, video_name


class VideoDataset_spatio_temporal_brightness(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, data_dir_3D, brightness, filename_path, transform, database_name, feature_type):
        super(VideoDataset_spatio_temporal_brightness, self).__init__()


        if database_name == 'my_train':
            column_names = ['name', 't1','t2','mos']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'my_test':
            column_names = ['name', 't1','t2','mos']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()


        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.database_name = database_name
        self.brightness = brightness

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if self.database_name == 'my_train' or self.database_name == 'my_test':
            video_name = self.video_names[idx]
            video_name_str = video_name.split('.')[0]

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)

        video_channel = 3
        video_height_crop = 512
        video_width_crop = 512
        video_length_read = 8


        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])
        j = 0
        for img in os.listdir(path_name):
            imge_name = os.path.join(path_name, img)
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[j] = read_frame
            j += 1

        # read temporal features

        if self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            brightness_folder_name = os.path.join(self.brightness, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048 + 256 + 144])
            for i in range(video_length_read):
                i_index = i

                feature_3D_slow = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()

                feature_3D_fast = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()

                brightness_consistency = np.load(
                    os.path.join(brightness_folder_name, 'brightness_consistency' + str(i_index) + '.npy')
                )
                brightness_consistency = torch.from_numpy(brightness_consistency)
                brightness_consistency = brightness_consistency.squeeze()
                brightness_consistency *= 10

                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast, brightness_consistency])
                transformed_feature[i] = feature_3D

        return transformed_video, transformed_feature, video_score, video_name
    

class DicomDataset_spatio_temporal_brightness(data.Dataset):
    """Read DICOM data for feature extraction"""

    def __init__(self, split,dcm_path,data_dir, data_dir_3D, brightness, filename_path, transform, database_name, feature_type,
                 rotation_angle=(0, 10), translation=(-30, 30)):
        super(DicomDataset_spatio_temporal_brightness, self).__init__()
        self.video_names = []  # 初始化视频名称列表
        if database_name == 'my_train' or database_name == 'my_test':
            column_names = ['name', 'mos']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()
        self.split = split 
        self.dcm_path = dcm_path
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.database_name = database_name
        self.brightness = brightness

        self.rotation_angle = rotation_angle
        self.translation = translation
        self.rotation_angle = (-70, 70)
        self.translation = (50, 150)

        self.augmentation_probability = 0.6  # 可以根据需求调整
        self.motion_artifact_probability=0.5
        self.metal_artifact_probability=0.3
        # with open("/home/data/shenhao/lung_quality/med/seg/"+split+"_data.txt", "rb") as fp:     #使用 with 语句打开文件 "split_data.txt",split可以是train，val,test,以二进制读取模式打开文件
        #     self.file_list = pickle.load(fp)            #从打开的fp文件中加载数据，并将加载的数据赋值给类的实例变量 self.file_list。
        #     #print(self.file_list,"list")                #打印 self.file_list 的内容。这行代码用于检查加载的数据，以确保正确读取了文件。
        # with open("/home/data/shenhao/lung_quality/med/seg/"+split+"_labels.txt", "rb") as fpl:
        #     self.labels = pickle.load(fpl)
        #     #print(self.labels,"labels")
        # self.size = len(self.file_list)                                                      #计算加载的数据集的大小，即 self.file_list 中的元素个数，
        

    def __len__(self):
        
        return self.length

    def __getitem__(self, idx):
        # fname = self.file_list[index]           #通过索引 index 获取 self.file_list 中对应位置的文件名， "xyz.nii"
        # fpath = os.path.join(self.data_dir, str(fname))
        if self.database_name == 'my_train' or self.database_name == 'my_test':
            video_name = self.video_names[idx]
            video_name_str = str(video_name)[1:-1]

            #print(f"video_name: {video_name}, type: {type(video_name)}")
            # video_name_str = video_name.split('.')[0]
            #video_name_str =  video_name[1:-1]
            

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)
        dcm_path = os.path.join(self.dcm_path, video_name_str)
        # print(video_name_str,'这是video_name')  ##video_name_str就是对于的文件名
        # print(dcm_path,'这是dcmpath')

        rotation_angle = 0
        three_D_rotation_angle=[0,0,0]
        translation = [0.0, 0.0, 0.0]
        motion_translation = [0.0, 0.0, 0.0]
        label3=0
        
        
        def load_metal_artifact_labels(file_path):
            # 读取 CSV 文件
            data = pd.read_csv(file_path, header=0)  # header=0 指定第一行作为列名
            # 去掉 'name' 列中的引号和多余的空格，确保匹配时的一致性
            # 创建字典，键为 name，值为 mos
            return dict(zip(data['name'],data['metal']))
        def load_motion_artifact_labels(file_path):
            # 读取 CSV 文件
            data = pd.read_csv(file_path, header=0)  # header=0 指定第一行作为列名
            # 去掉 'name' 列中的引号和多余的空格，确保匹配时的一致性
            
            # 创建字典，键为 name，值为 mos
            return dict(zip(data['name'],data['motion']))
        #DATAINFO_TEST = '/home/gem/sh/new_light_vqa/output/final_label_without_luna16_use_data_balance.csv'
        DATAINFO_TEST = '/home/gem/sh/new_light_vqa/csv_label/SZ_true_619data_with_true_label_3_31.csv'
        metal_artifact_labels = load_metal_artifact_labels(DATAINFO_TEST)
        motion_artifact_labels = load_motion_artifact_labels(DATAINFO_TEST)
        
        # print(metal_artifact_labels,'****这是metal_artifact_labels****')
        # print(motion_artifact_labels,'****这是motion_artifact_labels****')
        #folder_name = video_name_str.item()  # 获取文件夹名
        video_name_str_cleaned = video_name_str.strip("'")
        video_name_str_cleaned = f"'{video_name_str}'"
        #print(video_name_str_cleaned,'****这是video_name_str_cleaned')
        label4=0
        # 查找标签
        label4 = metal_artifact_labels.get(video_name_str_cleaned)  # 直接读取标签
        #label4 = metal_artifact_labels.get(video_name_str,0)  # 直接读取标签
        #print(label4,video_name_str,'****这是label4的值和对应的文件名****')
        label3 = motion_artifact_labels.get(video_name_str_cleaned)
        #print(label3,video_name_str,'****这是label3的值和对应的文件名****')
        
        image_array = np.zeros((1, 512, 512), dtype=np.float32)  # 默认形状为 (1, 512, 512)
        image_array, image3D = read_dicom_series_as_numpy_array(dcm_path)
        # if check_image_size(dcm_path):
        #     image_array, image3D = read_dicom_series_as_numpy_array(dcm_path)
            
        # else:
        #     print("DICOM 文件尺寸不符合要求，跳过")
        
        # print(image_array.shape,'shape,image array')

        if random.random() < self.augmentation_probability:
            rotation_angle = np.random.randint(self.rotation_angle[0], self.rotation_angle[1])
            three_D_rotation_angle=[0,0,rotation_angle]
            translation = [
                np.random.randint(self.translation[0], self.translation[1]),
                np.random.randint(self.translation[0], self.translation[1]),
                0
            ]
            #print(f"进行了旋转和平移，旋转角度: {rotation_angle}, 平移参数: {translation}")
            
            image_array, transformed_image = rotate_and_translate(image3D, three_D_rotation_angle, translation)
            
              # 直接读取标签
            # if random.random() < self.motion_artifact_probability and label3 == 0:
            #     random_y_translation = np.random.randint(0, 11)  # 在0到10之间生成一个随机整数作为y分量
            #     random_x_translation = np.random.randint(0, 11)
            #     random_z_translation = np.random.randint(0, 11)  
            #     motion_translation = [random_x_translation, random_y_translation,random_z_translation]  # x分量为0，y分量为随机数
            #     image_array = simulate_motion_blur(image_array, motion_translation)
            #     point = np.array([random_x_translation, random_y_translation,random_z_translation])
            #         # 计算点到原点的欧氏距离
            #     distance_to_origin = np.linalg.norm(point)
            #     #label3=distance_to_origin/10
            #     label3=1
            
            
        # if random.random() < self.metal_artifact_probability and label4 == 0:
        #     image_array = simulate_metal_artifact(image_array)
        #     label4=1

        #nii_path = os.path.join(dcm_path, f"{str(video_name_str).split('.')[0]}.nii")
        
        extracted_value = dcm_path.split('/')[-1]
       
        segment_path='/mnt/gemlab_data_2/User_database/shenhao/whole_sz_and_luna16/whole_sz_and_luna16_data'##用原来的分割先试一下
        #nii_path = os.path.join(dcm_path, f"{str(video_name_str).split('.')[0]}_segment.nii")
        segment_path = os.path.join(segment_path, f"{str(video_name_str)}")  ###2025.2.20自己加的
        
        nii_path = os.path.join(segment_path, f"{str(extracted_value)}_segment.nii")
        #nii_path = os.path.join(dcm_path, f"{str(video_name_str).split('.')[0]}_segment.nii")

        #print(nii_path,'这是nii_path')
        #nii_path=os.path.isfile(dicom_path) and dicom_file.lower().endswith('.nii')
        voxel_spacing = get_voxel_spacing(nii_path)
        voxel_size = np.array(voxel_spacing)
        nii_data, affine = load_nii_file(nii_path)
        original_volume = calculate_original_volume(nii_data, voxel_size)

        transformed_volume, transformed_mask = calculate_transformed_volume(nii_data, voxel_size, rotation_angle=rotation_angle,translation=translation)
        volume_percentage = calculate_volume_percentage(original_volume, transformed_volume)
        
        print(f"这是update训练的sh-原始体积:{original_volume / 1000},分割后的体积: {transformed_volume / 1000}，比率: {volume_percentage},文件名:{video_name_str},旋转参数:{rotation_angle}平移参数:{translation}")
        epsilon = 1e-10
        max_value = np.max(image_array)
        print(max_value,'这是 max_value')    #254 这是 max_value
        if max_value>=2900:
            label4=1
        else:
            label4=0
        image_array = np.array([resize(image, (512, 512), order=2) for image in image_array])
        image_array = (image_array-image_array.min())  / (image_array.max()  - image_array.min() + epsilon)         #归一化通过这个操作，图像的所有像素值都被映射到 [0, 1] 的范围内
        image_array = (image_array * 255).astype(np.uint8)
        
        if rotation_angle>0:
            rotation_angle=rotation_angle/360
        if rotation_angle<0:
            rotation_angle=rotation_angle/(-360)
        if rotation_angle==0:
            rotation_angle=0
        #label = int(self.labels[index])
        label1 = rotation_angle
        label2 = volume_percentage
        
        label1 = torch.FloatTensor(np.array(float(label1)))
        label2 = torch.FloatTensor(np.array(float(label2)))

        video_channel = 3
        video_height_crop = 512
        video_width_crop = 512
        video_length_read = 1    #####改完batch的话要来这里改，如果batch=1，则这里1，

        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])
        j = 0
        for image_i in range(len(image_array)):
            image = image_array[image_i]
            
            # 如果是灰度图像，将其转换为三通道图像
            if len(image.shape) == 2:  # 单通道灰度图像
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # 转换为三通道图像 (BGR)
                print('yes 是灰度图像')
            # 转换为 uint8 类型
            
            image = (image / np.max(image) * 255).astype(np.uint8)

            # 转换为 RGB 格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 转换为 PIL 图像并应用 transform
            read_frame = Image.fromarray(image_rgb)
            read_frame = self.transform(read_frame)
            transformed_video[j] = read_frame
            j += 1
            if j >= video_length_read:
                break

        # 读取时空亮度一致性和 3D 特征
        if self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            brightness_folder_name = os.path.join(self.brightness, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048 + 256 + 144])

            for i in range(video_length_read):
                i_index = i

                feature_3D_slow = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow).squeeze()

                feature_3D_fast = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast).squeeze()

                brightness_consistency = np.load(
                    os.path.join(brightness_folder_name, 'brightness_consistency' + str(i_index) + '.npy')
                )
                brightness_consistency = torch.from_numpy(brightness_consistency).squeeze()
                brightness_consistency *= 10

                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast, brightness_consistency])
                transformed_feature[i] = feature_3D

                print(transformed_video.shape,'transformed_video de shape')
        return transformed_video, transformed_feature, video_name, label1,label2,label3,label4
    

def read_dicom_series_as_numpy_array(file_path):
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(file_path)
    if not series_IDs:
        raise ValueError(f"在 {file_path} 中找不到 DICOM 系列")
    
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file_path, series_IDs[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3D = series_reader.Execute()
    image_array = sitk.GetArrayFromImage(image3D)
    
    return image_array, image3D
# import os
# import SimpleITK as sitk
# import numpy as np
# def check_image_size(dcm_path):
#     """
#     检查 DICOM 文件夹中的所有 DICOM 文件的尺寸是否为 (512, 512)
#     如果不是 512x512，返回 False，表示跳过。
#     """
#     dicom_files = [f for f in os.listdir(dcm_path) if f.endswith('.dcm')]
    
#     if not dicom_files:
#         raise ValueError(f"在 {dcm_path} 中找不到 DICOM 文件 (.dcm)")
    
#     # 读取第一个 DICOM 文件来检查尺寸
#     first_dcm_file = os.path.join(dcm_path, dicom_files[0])
#     image = sitk.ReadImage(first_dcm_file)
    
#     size = image.GetSize()  # 获取图像的尺寸 (宽度, 高度, 深度)
    
    
#     # 检查是否符合 (512, 512)
#     if size[0] != 512 or size[1] != 512:
#         print(f"跳过 {dcm_path}，因为尺寸不是 512x512")
#         return False  # 如果尺寸不是 512x512，返回 False
    
#     return True  # 如果尺寸符合要求，返回 True
# def read_dicom_series_as_numpy_array(file_path):
#     # 获取路径下所有的 .dcm 文件
#     dicom_files = [f for f in os.listdir(file_path) if f.endswith('.dcm')]
#     if not dicom_files:
#         raise ValueError(f"在 {file_path} 中找不到 DICOM 文件 (.dcm)")

#     valid_dicom_files = []
#     for dicom_file in dicom_files:
#         dicom_file_path = os.path.join(file_path, dicom_file)
#         # 读取单个 DICOM 文件
#         image = sitk.ReadImage(dicom_file_path)
#         size = image.GetSize()
#         if size[0] == 512 and size[1] == 512:
#             valid_dicom_files.append(dicom_file_path)

#     if not valid_dicom_files:
#         raise ValueError(f"在 {file_path} 中未找到尺寸为 512x512 的 DICOM 文件")

#     # 读取 DICOM 文件序列
#     series_reader = sitk.ImageSeriesReader()
#     series_reader.SetFileNames(valid_dicom_files)
#     # 读取图像数据
#     image3D = series_reader.Execute()
#     # 将 SimpleITK 图像转换为 NumPy 数组
#     image_array = sitk.GetArrayFromImage(image3D)
#     return image_array, image3D



def find_lung_center(image3D):
    # 假设 image3D 是一个包含多个通道的图像，使用 Cast 转换为单通道图像
    image3D = sitk.Cast(image3D, sitk.sitkInt16)  # 你可以根据需求选择合适的像素类型

    lung_mask = sitk.BinaryThreshold(image3D, lowerThreshold=-1024, upperThreshold=0, insideValue=1, outsideValue=0)
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(lung_mask)
    center_of_mass = label_shape_filter.GetCentroid(1)
    
    return center_of_mass

def rotate_and_translate(image3D, rotation_angles_deg, translation_vector):
    rotation_angles_rad = np.deg2rad(rotation_angles_deg)
    center_of_mass = find_lung_center(image3D)
    rotation_transform = sitk.Euler3DTransform()
    rotation_transform.SetCenter(center_of_mass)
    rotation_transform.SetRotation(rotation_angles_rad[0], rotation_angles_rad[1], rotation_angles_rad[2])
    rotated_image = sitk.Resample(image3D, image3D, rotation_transform, sitk.sitkLinear, -1024.0)
    translation_transform = sitk.TranslationTransform(3)
    translation_transform.SetOffset(translation_vector)
    transformed_image = sitk.Resample(rotated_image, image3D, translation_transform, sitk.sitkLinear, -1024.0)
    transformed_image_np = sitk.GetArrayFromImage(transformed_image)
    
    return transformed_image_np, transformed_image

def simulate_motion_blur(original_image_np, motion_translation_vector, blend_factor=0.5):
    motion_blurred_image_np = original_image_np.copy()
    for i in range(3):
        if motion_translation_vector[i] != 0:
            motion_blurred_image_np = np.roll(motion_blurred_image_np, int(motion_translation_vector[i]), axis=i)
    final_image_np = original_image_np * (1 - blend_factor) + motion_blurred_image_np * blend_factor
    
    return final_image_np


def add_metal_balls(image_array, metal_locations):
    max_intensity = np.max(image_array)
    for slice_idx, row, col in metal_locations:
        if 3 <= row < image_array.shape[1] - 3 and 3 <= col < image_array.shape[2] - 3:
            image_array[slice_idx, row, col] = max_intensity * 2.0  # 增加金属球的强度
    return image_array

def add_scatter_lines(image_array, metal_locations):
    for slice_idx, row, col in metal_locations:
        for i in range(100):  # 调整散射线的数量
            angle = np.random.rand() * np.pi * 2
            length = np.random.randint(50, 100)  # 调整散射线的长度
            for l in range(length):
                row_shift = int(np.sin(angle) * l)
                col_shift = int(np.cos(angle) * l)
                if 0 <= row + row_shift < image_array.shape[1] and 0 <= col + col_shift < image_array.shape[2]:
                    image_array[slice_idx, row + row_shift, col + col_shift] += 20000 / (i + 1)  # 调整散射强度
    return image_array


def simulate_metal_artifact(input_array):
    slices, rows, cols = input_array.shape
    num_metal_slices = max(1, slices // 5)
    metal_locations = [(np.random.randint(0, slices), np.random.randint(3, rows-3), np.random.randint(3, cols-3)) for _ in range(num_metal_slices)]
    image_array = input_array.copy()  # 复制输入数组，以免修改原始数据
    image_array = add_metal_balls(image_array, metal_locations)
    image_array = add_scatter_lines(image_array, metal_locations)
    
    theta = np.linspace(0., 180., max(rows, cols), endpoint=False)
    sinogram_slices = [radon(image_array[slice_idx], theta=theta, circle=False) for slice_idx in range(slices)]
    
    reconstructed_image = np.zeros_like(input_array)
    for slice_idx, sinogram in enumerate(sinogram_slices):
        reconstructed_slice = iradon(sinogram, theta=theta, circle=False)
        reconstructed_image[slice_idx] = resize(reconstructed_slice, (rows, cols), mode='reflect', anti_aliasing=True)
    
    # 全局直方图均衡化
    reconstructed_image = exposure.equalize_hist(reconstructed_image)
    reconstructed_image = (reconstructed_image * 65535).astype(np.uint16)
    
    return reconstructed_image

def get_voxel_spacing(file_path):
    nii_img = nib.load(file_path)
    header = nii_img.header
    voxel_spacing = header.get_zooms()
    return voxel_spacing

def load_nii_file(file_path):
    nii_img = nib.load(file_path)
    nii_data = nii_img.get_fdata()
    return nii_data, nii_img.affine

def calculate_original_volume(mask, voxel_size):
    return np.sum(mask) * np.prod(voxel_size)

def calculate_transformed_volume(mask, voxel_size, rotation_angle, translation):
    # 进行旋转
    rotated_mask = rotate(mask, angle=rotation_angle, reshape=False, mode='constant', cval=mask.min())

    # 进行平移
    translation_matrix = np.array([translation[0], translation[1],translation[2]])
    translated_mask = shift(rotated_mask, translation_matrix)

    # 计算旋转平移后的体积
    transformed_volume = np.sum(translated_mask) * np.prod(voxel_size)
    return transformed_volume, translated_mask

def calculate_volume_percentage(original_volume, translated_volume):
    return (translated_volume / original_volume)








