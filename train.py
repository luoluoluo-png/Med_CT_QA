
# -*- coding: utf-8 -*-   现在用的813修改后的
import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from utils import performance_fit
from utils import L1RankLoss
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
from scipy.stats import spearmanr, pearsonr
from my_dataloader_max_metal import VideoDataset_spatio_temporal_brightness, DicomDataset_spatio_temporal_brightness
from final_fusion_model_3ceng import swin_small_patch4_window7_224 as create_model
from torchvision import transforms
import time
from sklearn.metrics import mean_squared_error

def main(args):
    
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cuda:2")
    print(torch.__version__,'version')  # 查看 PyTorch 版本
    print(torch.cuda.is_available(),'检查是否支持 CUDA')  # 检查是否支持 CUDA
    print("Available GPUs:", torch.cuda.device_count())  # 查看可用 GPU 数量
    
    if torch.cuda.is_available():
        print("Current CUDA device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    model = create_model().to(device)
    # 使用 DataParallel 来并行化模型
    #model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])  # 指定使用 4 张卡

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=0.0002, weight_decay=0.0000001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)

    if args.loss_type == 'L1RankLoss':
        criterion = L1RankLoss(batchsize=args.train_batch_size)
    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss().to(device)

    param_num = 0
    for param in pg:
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    videos_dir = '/mnt/gemlab_data_2/User_database/shenhao/whole_sz_and_luna16/key_frames_2_21'  ##key_frames
    data_dir_3D = '/mnt/gemlab_data_2/User_database/shenhao/whole_sz_and_luna16/temporal_feature2_21/003493'  ###temporal_feature
    brightness = '/mnt/gemlab_data_2/User_database/shenhao/whole_sz_and_luna16/brightness_consistency_2_21' ##brightness_consistency
    dcm_path='/mnt/gemlab_data_2/User_database/shenhao/SZ_groundtruth/truth_motion_metal_data'  ##dcm_path
    
    datainfo_train = '/home/gem/sh/new_light_vqa/512_size/my_train_true.csv'
    datainfo_test = '/home/gem/sh/new_light_vqa/512_size/my_val_true.csv'
    
    # datainfo_train = '/home/gem/sh/new_light_vqa/update/output/3_8/my_train_little.csv'
    # datainfo_test = '/home/gem/sh/new_light_vqa/update/output/3_8/my_val_little.csv'
    
    # datainfo_train = '/home/gem/sh/new_light_vqa/update/my_train_4.csv'
    # datainfo_test = '/home/gem/sh/new_light_vqa/update/my_val_4.csv'
    
    # transformations_train = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize(mean=[0.204, 0.204, 0.204], std=[0.327, 0.327, 0.327])])
    # transformations_test = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize(mean=[0.204, 0.204, 0.204], std=[0.327, 0.327, 0.327])])
    
    transformations_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.804, 0.804, 0.804], std=[0.927, 0.927, 0.927])])
    transformations_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.804, 0.804, 0.804], std=[0.927, 0.927, 0.927])])
    
    split = 'train'
    trainset = DicomDataset_spatio_temporal_brightness(split,  dcm_path, videos_dir, data_dir_3D, brightness, datainfo_train, transformations_train, 'my_train', 'SlowFast')
    testset = DicomDataset_spatio_temporal_brightness(split, dcm_path, videos_dir, data_dir_3D, brightness, datainfo_test, transformations_test, 'my_test', 'SlowFast')

    ## dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size,
                                               shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=args.num_workers)

    best_test_criterion = 0.76  # SROCC min
    best_test = []

    # 打开文件用于保存训练和验证结果
    results_file = os.path.join(args.results_path, '4_6_training_results.txt')
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    # with open(results_file, 'w') as f:
    #     f.write("Epoch\tTrain_Loss\tVal_SRCC1\tVal_PLCC1\tVal_RMSE1\tVal_SRCC2\tVal_PLCC2\tVal_RMSE2\tVal_SRCC3\tVal_PLCC3\tVal_RMSE3\tVal_Accuracy4\tVal_F1_4\tVal_Recall4\tVal_Precision4\n")

    print('Starting training:')

    old_save_name = None

    for epoch in range(args.epochs):
        model.train()
        batch_losses = []
        batch_losses_each_disp = []
        session_start_time = time.time()
        for i, (video, tem_f, _, target1, target2, target3, target4) in enumerate(train_loader):

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
            optimizer.zero_grad()
            ######原来的
            target1 = target1.expand(8).unsqueeze(1)
            target2 = target2.expand(8).unsqueeze(1)
            #target3 = target3.expand(8).unsqueeze(1)   ##2024.11.11把回归任务变成2分类
            # target3 = target3.repeat(8)
            # target4 = target4.repeat(8)
            ########
            # print(outputs[2].shape,'outputs[2].shape')
            # print(target3.shape,'target3.shape')
            # print(outputs[3].shape,'outputs[3].shape')
            # print(target3.shape,'target4.shape')
            #####2025.1.5修改的
            # 获取当前批次大小
            # 获取当前的批次
            target3 = target3
            target4 = target4

            #####
            

            loss1 = criterion1(outputs[0], target1)
            loss2 = criterion1(outputs[1], target2)
            loss3 = criterion2(outputs[2], target3)
            loss4 = criterion2(outputs[3], target4)
            print(loss1,'loss1')
            print(loss2,'loss2')
            print(loss3,'loss3')
            print(loss4,'loss4')
            loss = loss1 + loss2 + loss3 + loss4
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())
            loss.backward()

            optimizer.step()

            if (i + 1) % (args.print_samples // args.train_batch_size) == 0:
                session_end_time = time.time()
                avg_loss_epoch = sum(batch_losses_each_disp) / (args.print_samples // args.train_batch_size)
                print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' % \
                      (epoch + 1, args.epochs, i + 1, len(trainset) // args.train_batch_size, \
                       avg_loss_epoch))
                batch_losses_each_disp = []
                print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                session_start_time = time.time()

        avg_loss = sum(batch_losses) / (len(trainset) // args.train_batch_size)
        print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

        scheduler.step()
        lr = scheduler.get_last_lr()
        print('The current learning rate is {:.06f}'.format(lr[0]))

        # do validation after each epoch
        targets1 = []
        predictions1 = []
        targets2 = []
        predictions2 = []
        targets3 = []
        predictions3 = []
        targets4 = []
        predictions4 = []
        with torch.no_grad():
            model.eval()
            label1 = np.zeros([len(testset)])
            y_output1 = np.zeros([len(testset)])
            
            correct3 = 0
            all_pred3s = []
            all_label3s = []
            correct4 = 0
            all_pred4s = []
            all_label4s = []
            
            label2 = np.zeros([len(testset)])
            y_output2 = np.zeros([len(testset)])
            
            for i, (video, tem_f, _, target1, target2, target3, target4) in enumerate(test_loader):
                video = video.to(device)
                tem_f = tem_f.to(device)
                video = torch.reshape(video, [video.shape[0] * video.shape[1], 3, 512, 512])
                tem_f = torch.reshape(tem_f, [tem_f.shape[0] * tem_f.shape[1], 2304 + 144])
                label1[i] = target1.item()
                label2[i] = target2.item()

                target1 = target1.to(device, dtype=torch.float32)
                target2 = target2.to(device, dtype=torch.float32)
                #target3 = target3.to(device, dtype=torch.float32)
                target3 = target3.to(device, dtype=torch.long)
                target4 = target4.to(device, dtype=torch.long)
                print(video.shape,'video shape')
                print(tem_f.shape,'tem_f shape')
                outputs = model(video, tem_f)
                
                #pred3 = (torch.sigmoid(outputs[2]) > 0.5).float()
                pred3 = outputs[2].argmax(dim = 1)
                
                correct3 += pred3.eq(target3.view_as(pred3)).sum().item()
                

                # 保存预测结果和真实标签
                all_pred3s.append(pred3.cpu().numpy())  # 转移到CPU
                all_label3s.append(target3.cpu().numpy())

                pred4 = outputs[3].argmax(dim = 1)
                correct4+= pred4.eq(target4.view_as(pred4)).sum().item()

                # 保存预测结果和真实标签
                all_pred4s.append(pred4.cpu().numpy())  # 转移到CPU
                all_label4s.append(target4.cpu().numpy())

                y_output1[i] = torch.mean(outputs[0]).item()
                y_output2[i] = torch.mean(outputs[1]).item()
                
                targets1.extend(target1.cpu().numpy().reshape(-1))  # 添加了reshape
                predictions1.extend(torch.mean(outputs[0]).detach().cpu().numpy().reshape(-1))  # 添加了reshape
                targets2.extend(target2.cpu().numpy().reshape(-1))  # 添加了reshape
                predictions2.extend(torch.mean(outputs[1]).detach().cpu().numpy().reshape(-1))  # 添加了reshape
                # targets3.extend(target3.cpu().numpy().reshape(-1))  # 添加了reshape
                # predictions3.extend(torch.mean(outputs[2]).detach().cpu().numpy().reshape(-1))  # 添加了reshape
                targets3.extend(target3.cpu().numpy())
                predictions3.append(torch.mean(outputs[2], dim=0).detach().cpu().numpy())
                
                targets4.extend(target4.cpu().numpy())
                predictions4.append(torch.mean(outputs[3], dim=0).detach().cpu().numpy())

                predictions3_np = np.array(predictions3)
                predictions4_np = np.array(predictions4)
                # 将概率转换为类别标签
                threshold = 0.5
                predictions3_labels = (predictions3_np[:, 1] > threshold).astype(int)  # 假设你希望第二列的概率大于阈值时预测为正类别
                predictions4_labels = (predictions4_np[:, 1] > threshold).astype(int)  # 假设你希望第二列的概率大于阈值时预测为正类别
            #######
            test_PLCC1, test_SRCC1, test_KRCC1, test_RMSE1 = performance_fit(label1, y_output1)
            test_PLCC2, test_SRCC2, test_KRCC2, test_RMSE2 = performance_fit(label2, y_output2)
            #######
            
            ########### 计算准确率
            new_acc3 = correct3 / len(test_loader.dataset)
            new_acc4 = correct4 / len(test_loader.dataset)
            # 将所有预测结果和标签拼接成一个大数组
            all_pred3s = np.concatenate(all_pred3s)
            all_label3s = np.concatenate(all_label3s)

            all_pred4s = np.concatenate(all_pred4s)
            all_label4s = np.concatenate(all_label4s)
            
            # 计算 AUC
            try:
                new_auc3 = roc_auc_score(all_label3s, all_pred3s)
                new_auc4 = roc_auc_score(all_label4s, all_pred4s)  # AUC 计算时，pred是概率值
            except ValueError:
                new_auc3 = 0.0  # 如果没有正负样本，AUC无法计算，返回0
                new_auc4 = 0.0 
            # 计算 F1-score
            new_f1_3 = f1_score(all_label3s, all_pred3s)
            new_f1_4 = f1_score(all_label4s, all_pred4s)

            # 计算 Precision
            new_precision3 = precision_score(all_label3s, all_pred3s)
            new_precision4 = precision_score(all_label4s, all_pred4s)
            # 计算 Recall
            new_recall3 = recall_score(all_label3s, all_pred3s)
            new_recall4 = recall_score(all_label4s, all_pred4s)
            ###############
            
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
            print(targets3,'targets3')
            print(predictions3_labels,'predictions3_labels')
            # 计算AUC之前，先检查是否有两个类别
            if len(set(targets3)) > 1:  # 检查是否有多个类别
                val_auc3 = roc_auc_score(targets3, predictions3_labels)
            else:
                print("Warning: Only one class present in targets3, skipping AUC calculation.")
                val_auc3 = None  # 或者使用其他默认值
            print(targets4,'targets4')
            print(predictions4_labels,'predictions4_labels')

            
            val_accuracy4 = accuracy_score(targets4, predictions4_labels)
            val_f1_4 = f1_score(targets4, predictions4_labels)
            val_recall4 = recall_score(targets4, predictions4_labels)
            val_precision4 = precision_score(targets4, predictions4_labels)
            #val_auc4 = roc_auc_score(targets4, predictions4_labels)
            if len(set(targets4)) > 1:  # 检查是否有多个类别
                val_auc4 = roc_auc_score(targets4, predictions4_labels)
            else:
                print("Warning: Only one class present in targets4, skipping AUC calculation.")
                val_auc4 = None  # 或者使用其他默认值
            # print(
            #     'Epoch {} completed. The result on the test database: \
            #         val_srcc1: {:.4f}, val_plcc1: {:.4f}, val_srcc2: {:.4f},\
            #         val_accuracy3: {:.4f}, val_f1_3: {:.4f}, \
            #         val_recall3: {:.4f}, and val_precision3: {:.4f}, val_accuracy4: {:.4f}, val_f1_4: {:.4f}, \
            #         val_recall4: {:.4f}, and val_precision4: {:.4f}'.format(
            #         epoch + 1, \
            #         val_srcc1, val_plcc1, val_srcc2, val_plcc2, val_srcc3, val_plcc3, \
            #         val_accuracy4, val_f1_4, val_recall4, val_precision4))
            # 计算 avg_loss 时，如果出现 None 就设置默认值
            avg_loss = avg_loss if avg_loss is not None else 0.0  # 设置默认值为 0.0
                        
            result_str = (
                f"{epoch + 1}\t{avg_loss:.4f}\n"
                f"SRCC1: {val_srcc1:.4f}, PLCC1: {val_plcc1:.4f}, RMSE1: {val_rmse1:.4f}\n"
                f"SRCC2: {val_srcc2:.4f}, PLCC2: {val_plcc2:.4f}, RMSE2: {val_rmse2:.4f}\n"
                
                f"test_SRCC1: {test_SRCC1:.4f}, test_PLCC1: {test_PLCC1:.4f}, test_RMSE1: {test_RMSE1:.4f}, test_KRCC1: {test_KRCC1:.4f}\n"
                f"test_SRCC2: {test_SRCC2:.4f}, test_PLCC2: {test_PLCC2:.4f}, test_RMSE2: {test_RMSE2:.4f}, test_KRCC2: {test_KRCC2:.4f}\n\n"
                
                f"Accuracy3: {val_accuracy3:.4f}, F1_3: {val_f1_3:.4f}, Recall3: {val_recall3:.4f}, Val_auc3: {val_auc3:.4f}, Precision3: {val_precision3:.4f}\n"
                f"new_Accuracy3: {new_acc3:.4f}, new_F1_3: {new_f1_3:.4f}, new_Recall3: {new_recall3:.4f}, new_Val_auc3: {new_auc3:.4f}, new_Precision3: {new_precision3:.4f}\n"
                
                f"Accuracy4: {val_accuracy4:.4f}, F1_4: {val_f1_4:.4f}, Recall4: {val_recall4:.4f}, Val_auc4: {val_auc4:.4f}, Precision4: {val_precision4:.4f}\n"
                f"new_Accuracy4: {new_acc4:.4f}, new_F1_4: {new_f1_4:.4f}, new_Recall4: {new_recall4:.4f}, new_Val_auc4: {new_auc4:.4f}, new_Precision4: {new_precision4:.4f}\n\n"
            )

            # 将结果写入文件
            with open(results_file, 'a') as f:
                f.write(result_str)
                
            
            # 将结果写入文件
            # with open(results_file, 'a') as f:
            #     f.write(f"{epoch + 1}\t{avg_loss:.4f}\t{val_srcc1:.4f}\t{val_plcc1:.4f}\t{val_rmse1:.4f}\t{val_srcc2:.4f}\t{val_plcc2:.4f}\t{val_rmse2:.4f}\t{val_srcc3:.4f}\t{val_plcc3:.4f}\t{val_rmse3:.4f}\t{val_accuracy4:.4f}\t{val_f1_4:.4f}\t{val_recall4:.4f}\t{val_precision4:.4f}\n")

            # if val_auc4> best_test_criterion:
            #     print("Update best model using best_test_criterion in epoch {}".format(epoch + 1))
            #     best_test_criterion = val_auc4
            #     best_test = [val_srcc1, val_plcc1,val_rmse1, val_srcc2, val_plcc2,val_rmse2, val_accuracy3, val_f1_3, val_recall3, val_precision3,val_auc3, \
            #         val_accuracy4, val_f1_4, val_recall4, val_precision4,val_auc4]
            #     print('Saving model...')
            #     if not os.path.exists(args.ckpt_path):
            #         os.makedirs(args.ckpt_path)

            #     save_model_name = args.ckpt_path + '/' + 'last2_SI+TI_epoch_%d_val_auc4_%f.pth' % (epoch + 1, val_auc4)
            #     torch.save(model.state_dict(), save_model_name)
            #     # 如果需要保存整个模型，可以使用下面的代码
            #     torch.save(model, save_model_name.replace('.pth', '_full.pth'))
                
            #     old_save_name = save_model_name
            # 计算 AUC3 和 AUC4 的平均值
            avg_auc = (val_auc3 + val_auc4) / 2

            # 使用 AUC3 和 AUC4 的平均值来决定是否更新最佳模型
            if avg_auc > best_test_criterion:
                print("Update best model using avg_auc (AUC3 and AUC4) in epoch {}".format(epoch + 1))
                best_test_criterion = 0.7  # 更新最佳标准
                best_test = [val_srcc1, val_plcc1, val_rmse1, val_srcc2, val_plcc2, val_rmse2, 
                            val_accuracy3, val_f1_3, val_recall3, val_precision3, val_auc3, 
                            val_accuracy4, val_f1_4, val_recall4, val_precision4, val_auc4]
                print('Saving model...')
                
                # 如果保存路径不存在则创建
                if not os.path.exists(args.ckpt_path):
                    os.makedirs(args.ckpt_path)

                # 保存模型的路径
                save_model_name = args.ckpt_path + '/' + 'last2_SI+TI_epoch_%d_avg_auc_%f.pth' % (epoch + 1, avg_auc)
                
                # 保存模型
                torch.save(model.state_dict(), save_model_name)
                # 如果需要保存整个模型，可以使用下面的代码
                #torch.save(model, save_model_name.replace('.pth', '_full.pth'))
                
                old_save_name = save_model_name


    print('Training completed.')
    print(
        'The best training result on the test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            best_test[0], best_test[1], best_test[2], best_test[3]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str)
    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)
    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--n_trial', type=int, default=0)
    parser.add_argument('--results_path', type=str, default='/home/gem/sh/new_light_vqa/512_size/output/4_6', help='Path to save the results.')
    parser.add_argument('--exp_version', type=int)
    parser.add_argument('--print_samples', type=int, default=1000)
    parser.add_argument('--train_batch_size', type=int, default=1) ###batch
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=40)
    # misc
    parser.add_argument('--ckpt_path', type=str, default='/home/gem/sh/new_light_vqa/512_size/output/4_6')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--loss_type', type=str, default='L1RankLoss')

    parser.add_argument('--weights', type=str, default='/home/gem/sh/new_light_vqa/swin_small_patch4_window7_224.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=True)

    args = parser.parse_args()
    main(args)
