import csv
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import random
from utils.utils import write_csv, write_json, draw_ROC
from torchnet import meter
from utils.log_and_save import Logger, progress_bar
from utils.file_ops import *
from sklearn.metrics import roc_curve, roc_auc_score
from utils.train_utils import maybe_to_torch, to_cuda
from loss.loss_fn import FocalLoss, minentropyloss, LabelSmoothing
import argparse


from config.config_mvi_onestream import * 
from loader.mvi_dataset_onestream import MVI_Dataloader_mvi
from models.resnet3d_onestream import *  # triplet is here


import warnings
warnings.filterwarnings("ignore")


def train(Basic_Args):

    model_save_path = Basic_Args.save_path
    maybe_mkdir_p(model_save_path)

    train_dataloader = MVI_Dataloader_mvi(Basic_Args, phase='train', balance=True)
    test_dataloader = MVI_Dataloader_mvi(Basic_Args, phase='test', balance=False)

    if Basic_Args.use_attr:
        n_input_channels = 6
    else:
        n_input_channels = 2
    model = generate_model(model_depth=10, n_input_channels=n_input_channels, n_classes=2, args=Basic_Args)
    model.cuda()

    # criterion
    weight = torch.Tensor([1, 1]).cuda()
    focal_loss = FocalLoss(gamma=2, alpha=None)
    ce_loss = torch.nn.CrossEntropyLoss(weight)
    entropy_loss = minentropyloss

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=3e-5)
    
    # iter为单位
    loss_meter = meter.AverageValueMeter() 
    # epoch为单位
    epoch_loss = meter.AverageValueMeter()
    train_cm = meter.ConfusionMeter(2)
    train_AUC = meter.AUCMeter()
    train_AP = meter.APMeter()
    train_mAP = meter.mAPMeter()

    previous_best_variable = 0
    best_epoch = 0
    test_previous_best_variable = 0
    test_best_epoch = 0
    variable = 'auc'

    cap1_list, cap2_list, cap3_list = None, None, None
    fen1_list, fen2_list, fen3_list = None, None, None
    num_list, radio_list, have_ace_or_not_list = None, None, None

    # start train
    for epoch in range(Basic_Args.max_epoch):
        model.train()
        print(" =============================================== ")
        print('** Epoch %d' % (epoch))

        # epoch为单位的评估
        epoch_loss.reset()
        train_cm.reset()
        train_AUC.reset()
        train_AP.reset()
        train_mAP.reset()

        epoch_finish_flag = False

        # for i, data_dict in tqdm(enumerate(train_dataloader)):
        for k in range(Basic_Args.max_iter):
            if epoch_finish_flag:
                break

            loss_meter.reset()
            data_dict = next(train_dataloader)

            all_data = data_dict['all_data']
            label = data_dict['label']
            
            resize_factor = data_dict['resize_factor']
            cap1_list = data_dict['cap1_list']
            cap2_list = data_dict['cap2_list']
            fen1_list = data_dict['fen1_list']
            fen2_list = data_dict['fen2_list']

            num_list = data_dict['num_list']
            radio_list = data_dict['radio_list']
            have_ace_or_not_list = data_dict['have_ace_or_not_list']

            if Basic_Args.use_ourtriplet:
                magin_array = data_dict['magin_array'].cuda() # triplet

            sample_finish_flag = data_dict['sample_finish_flag']
            if sample_finish_flag:
                epoch_finish_flag = True
            else:
                epoch_finish_flag = False

            all_data = maybe_to_torch(all_data)
            label = maybe_to_torch(label)
            label = label.long() #long

            if torch.cuda.is_available():
                all_data = to_cuda(all_data)
                label = to_cuda(label)


            score = model(all_data, resize_factor, cap1_list, cap2_list, cap3_list, fen1_list, fen2_list, fen3_list, num_list, radio_list, have_ace_or_not_list)

            if Basic_Args.use_ourtriplet:
                triplet_loss = model.get_triplet_loss(label, magin_array, device= magin_array.device, type='batch_hard') # batch_all
            else:
                triplet_loss = 0.0

            optimizer.zero_grad()

            loss1 = focal_loss(score, label)
            loss2 = ce_loss(score, label)
            loss3 = entropy_loss(score)
            weight1, weight2, weight3 = 0, 1, 0
            loss = weight1 * loss1 + weight2 * loss2 + weight3 * loss3 + triplet_loss

            loss.backward()
            optimizer.step()

            if Basic_Args.use_ourtriplet:
                progress_bar(k, len(train_dataloader), 'total loss:{}, ce_loss:{}, triplet_loss:{}'.format(round(loss.item(), 4), round(loss2.item(), 4), round(triplet_loss.item(), 4)))
            else:
                progress_bar(k, len(train_dataloader), 'total loss:{}, ce_loss:{}'.format(round(loss.item(), 4), round(loss2.item(), 4)))

            loss_meter.add(loss.item())
            epoch_loss.add(loss.item())
            
            score_softmax = F.softmax(score, dim=1)
            train_cm.add(score_softmax.data, label.data)

            positive_score = torch.tensor([item[1] for item in score_softmax.data.cpu().numpy().tolist()])
            train_AUC.add(np.array(positive_score), np.array(label.data.cpu()))
            
            one_hot = torch.zeros(label.size(0), 2).scatter_(1, label.data.cpu().unsqueeze(1), 1)
            train_AP.add(score_softmax.data, one_hot)
            train_mAP.add(score_softmax.data, one_hot)

        train_se = [train_cm.value()[0][0] / (train_cm.value()[0][0] + train_cm.value()[0][1]),
                    train_cm.value()[1][1] / (train_cm.value()[1][0] + train_cm.value()[1][1])]

        print('Epoch {}, training se: {}'.format(epoch, train_se))

        # start validate
        model.eval()
        val_cm, val_AUC, val_AP, val_mAP, val_accuracy = valid(model, test_dataloader, Basic_Args)
        current_variable = val_AUC[0]

        if current_variable > previous_best_variable:
            previous_best_variable = current_variable
            best_epoch = epoch
            model.save(os.path.join(Basic_Args.save_path, 'model_best.pth'))
        
        if epoch % 5 == 0:
            model.save(os.path.join(Basic_Args.save_path, 'model_latest.pth'))

        print("LR: {}".format(optimizer.param_groups[0]['lr']))

        print(" ======== metric on valid set for epoch {} ======== ".format(epoch))
        print('best epoch:  {} and best variable: {}'.format(best_epoch, previous_best_variable))
        print('train_cm:  \n {}'.format(train_cm.value()))
        print('train_AUC: {}'.format(train_AUC.value()[0]))
        
        val_se = [val_cm[0][0] / (val_cm[0][0] + val_cm[0][1]),
                  val_cm[1][1] / (val_cm[1][0] + val_cm[1][1])]
        print('val_cm: \n  {}'.format(val_cm))
        print('val_AUC: {}, val_accuracy:{}, val_mAP:{}, val_AP:{}'.format(val_AUC[0], val_accuracy, val_mAP, val_AP))
        print('valid se: {}'.format(val_se))
        print(" ")


    model.save(os.path.join(Basic_Args.save_path, 'model_final.pth'))
    print("train finish on max_epoch: {}, Best model from Epoch: {}".format(Basic_Args.max_epoch, best_epoch))
    print('==================== final model ====================')
    test_dataloader = MVI_Dataloader_mvi(Basic_Args, phase='test', balance=False)
    onlinetest(model, test_dataloader, 'final', Basic_Args)
    print(" ")

    print('==================== best model ====================')
    model = generate_model(model_depth=10, n_input_channels=n_input_channels, n_classes=2, args=Basic_Args)
    model.load(os.path.join(Basic_Args.save_path, 'model_best.pth'))
    model.cuda()
    model.eval()
    onlinetest(model, test_dataloader, 'best', Basic_Args)
    print(" ")

    print('==================== last model ====================')
    model = generate_model(model_depth=10, n_input_channels=n_input_channels, n_classes=2, args=Basic_Args)
    model.load(os.path.join(Basic_Args.save_path, 'model_latest.pth'))
    model.cuda()
    model.eval()
    onlinetest(model, test_dataloader, 'latest', Basic_Args)



def valid(model, dataloader, Basic_Args):
    ce_loss = torch.nn.CrossEntropyLoss()
    # epoch为单位
    val_loss = meter.AverageValueMeter()
    val_cm = meter.ConfusionMeter(2)
    val_AUC = meter.AUCMeter()
    val_AP = meter.APMeter()
    val_mAP = meter.mAPMeter()

    val_loss.reset()
    val_cm.reset()
    val_AUC.reset()
    val_AP.reset()
    val_mAP.reset()
    
    # for auc
    y_true, y_scores = [], []
    correct = 0.0
    epoch_finish_flag = False

    cap1_list, cap2_list, cap3_list = None, None, None
    fen1_list, fen2_list, fen3_list = None, None, None
    num_list, radio_list, have_ace_or_not_list = None, None, None

    error_list = []
    with torch.no_grad():
        for k in range(Basic_Args.max_iter):
            if epoch_finish_flag:
                break
            data_dict = next(dataloader)

            all_data = data_dict['all_data']
            label = data_dict['label']
            sample_finish_flag = data_dict['sample_finish_flag']
            resize_factor = data_dict['resize_factor']
            name_list = data_dict['name_list']

            if Basic_Args.use_bio_marker:
                cap1_list = data_dict['cap1_list']
                cap2_list = data_dict['cap2_list']

                fen1_list = data_dict['fen1_list']
                fen2_list = data_dict['fen2_list']

                num_list = data_dict['num_list']
                radio_list = data_dict['radio_list']
                have_ace_or_not_list = data_dict['have_ace_or_not_list']

            if sample_finish_flag:
                epoch_finish_flag = True
            else:
                epoch_finish_flag = False

            all_data = maybe_to_torch(all_data)
            label = maybe_to_torch(label)
            label = label.long() 

            if torch.cuda.is_available():
                all_data = to_cuda(all_data)
                label = to_cuda(label)

            score = model(all_data, resize_factor, cap1_list, cap2_list, cap3_list, fen1_list, fen2_list, fen3_list, num_list, radio_list, have_ace_or_not_list)
                

            loss = ce_loss(score, label)
            val_loss.add(loss.item())

            score_softmax = F.softmax(score, dim=1)

            pred = score_softmax.data.max(1)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
            positive_score = [item[1] for item in score_softmax.data.cpu().numpy().tolist()]
            
            y_true.extend(label.data.cpu().numpy().tolist())
            y_scores.extend(positive_score)
            
            val_cm.add(score_softmax.data, label.data)
            val_AUC.add(np.array(positive_score), np.array(label.data.cpu()))
            
            one_hot = torch.zeros(label.size(0), 2).scatter_(1, label.data.cpu().unsqueeze(1), 1)
            val_AP.add(score_softmax.data, one_hot)
            val_mAP.add(score_softmax.data, one_hot)

            error_list += list(name_list[np.array(~pred.eq(label.data.view_as(pred)).cpu())])

    SKL_FPR, SKL_TPR, SKL_Thresholds = roc_curve(y_true, y_scores)
    SKL_AUC = roc_auc_score(np.array(y_true), np.array(y_scores), average='weighted')

    val_cm = val_cm.value()
    val_accuracy = sum([val_cm[c][c] for c in range(2)]) / val_cm.sum()

    val_sp = [(val_cm.sum() - val_cm.sum(0)[i] - val_cm.sum(1)[i] + val_cm[i][i]) / (val_cm.sum() - val_cm.sum(1)[i])
              for i in range(2)]

    val_se = [val_cm[i][i] / val_cm.sum(1)[i] for i in range(2)]
    val_se2 = [val_cm[0][0] / (val_cm[0][0] + val_cm[0][1]),
               val_cm[1][1] / (val_cm[1][0] + val_cm[1][1])]

    val_AUC = val_AUC.value()
    val_AP = val_AP.value()
    val_mAP = val_mAP.value()

    print('==> pred_error_num: ', len(error_list), ' id: ', error_list)
    return val_cm, val_AUC, val_AP, val_mAP, val_accuracy, 



def onlinetest(model, dataloader, note, Basic_Args):
    # =========================================== Prepare Metrics =====================================
    test_cm = meter.ConfusionMeter(2)
    test_AUC = meter.AUCMeter()
    test_AP = meter.APMeter()
    test_mAP = meter.mAPMeter()

    test_cm.reset()
    test_AUC.reset()
    test_AP.reset()
    test_mAP.reset()
    
    results = []
    correct = 0.0
    y_true, y_scores = [], []
    epoch_finish_flag = False

    cap1_list, cap2_list, cap3_list = None, None, None
    fen1_list, fen2_list, fen3_list = None, None, None
    num_list, radio_list, have_ace_or_not_list = None, None, None
    # =========================================== Test ============================================
    with torch.no_grad():
        for k in range(Basic_Args.max_iter):
            if epoch_finish_flag:
                break
            data_dict = next(dataloader)

            all_data = data_dict['all_data']
            label = data_dict['label']
            sample_finish_flag = data_dict['sample_finish_flag']
            resize_factor = data_dict['resize_factor']
            name_list = data_dict['name_list']

            if Basic_Args.use_bio_marker:
                cap1_list = data_dict['cap1_list']
                cap2_list = data_dict['cap2_list']

                fen1_list = data_dict['fen1_list']
                fen2_list = data_dict['fen2_list']

                num_list = data_dict['num_list']
                radio_list = data_dict['radio_list']
                have_ace_or_not_list = data_dict['have_ace_or_not_list']

            if sample_finish_flag:
                epoch_finish_flag = True
            else:
                epoch_finish_flag = False

            all_data = maybe_to_torch(all_data)
            label = maybe_to_torch(label)
            label = label.long()

            if torch.cuda.is_available():
                all_data = to_cuda(all_data)
                label = to_cuda(label)

            score = model(all_data, resize_factor, cap1_list, cap2_list, cap3_list, fen1_list, fen2_list, fen3_list, num_list, radio_list, have_ace_or_not_list)
            score_softmax = F.softmax(score, dim=1)

            # *************************** confusion matrix and AUC *************************
            pred = score_softmax.data.max(1)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
            positive_score = [item[1] for item in score_softmax.data.cpu().numpy().tolist()]
            
            y_true.extend(label.data.cpu().numpy().tolist())  # 用于sklearn计算AUC和ROC
            y_scores.extend(positive_score)
            
            test_cm.add(score_softmax.data, label.data)
            test_AUC.add(np.array(positive_score), np.array(label.data.cpu()))
            
            one_hot = torch.zeros(label.size(0), 2).scatter_(1, label.data.cpu().unsqueeze(1), 1)
            test_AP.add(score_softmax.data, one_hot)
            test_mAP.add(score_softmax.data, one_hot)

            # ******************************** record prediction results ******************************
            for l, p, ip in zip(label.detach(), score_softmax.detach(), name_list):
                if p[1] < 0.5:
                    results.append((ip, int(l), 0, round(float(p[0]), 4), round(float(p[1]), 4)))
                else:
                    results.append((ip, int(l), 1, round(float(p[0]), 4), round(float(p[1]), 4)))

    # ************************** TPR, FPR, AUC ******************************
    SKL_FPR, SKL_TPR, SKL_Thresholds = roc_curve(y_true, y_scores)
    SKL_AUC = roc_auc_score(np.array(y_true), np.array(y_scores), average='weighted')
    TNet_AUC, TNet_TPR, TNet_FPR = test_AUC.value()

    # ******************** Best SE, SP, Thresh, Matrix ***********************
    test_cm = test_cm.value()
    test_sp = [(test_cm.sum() - test_cm.sum(0)[i] - test_cm.sum(1)[i] + test_cm[i][i]) / (test_cm.sum() - test_cm.sum(1)[i])
              for i in range(2)]
    test_AP = test_AP.value()
    test_mAP = test_mAP.value()
    # *********************** accuracy and sensitivity ***********************
    test_accuracy = 100. * sum([test_cm[c][c] for c in range(2)]) / np.sum(test_cm)
    test_se = [100. * test_cm[i][i] / np.sum(test_cm[i]) for i in range(2)]

    print('test_acc: {}'.format(test_accuracy))
    print('test_avgse: {}, train_se0: {}, train_se1: {} '.format(round(np.average(test_se), 4), round(test_se[0], 4), round(test_se[1], 4)))
    print('SKL_AUC: {}, TNet_AUC: {}'.format(SKL_AUC, TNet_AUC)) # 两种AUC的计算方式
    print('val_mAP:{}, val_AP:{}'.format(test_mAP, test_AP))
    print('test_cm: \n {}'.format(test_cm))







if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='', help='raw_data_path')
    parser.add_argument('--save_path', type=str, default='', help='where to save')

    parser.add_argument('--use_attr', action='store_true')
    parser.add_argument('--use_ourtriplet', action='store_true')
    parser.add_argument('--use_bio_marker', action='store_true') # for test
    
    parser.add_argument('--split_prop', type=float, default=0.7)
    parser.add_argument('--max_epoch', type=int, default=60)
    parser.add_argument('--max_iter', type=int, default=100000)
    
    args = parser.parse_args()

    train(args)
