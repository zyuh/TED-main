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
import copy
import time
import shutil
# from config.config_mvi_onestream_test import * 
from loader.mvi_dataset_onestream_test import MVI_Dataloader_mvi_test
from models.resnet3d_onestream import *  # triplet is here


import warnings
warnings.filterwarnings("ignore")


def test(Basic_Args):
    not_exists_flag = True
    while not_exists_flag:
        if os.path.exists(os.path.join(Basic_Args.data_path, 'attr_biomarkers.csv')):
            sample_list = os.listdir(Basic_Args.data_path)
            sample_list = [each_sample for each_sample in sample_list if 'csv' not in each_sample]
            filename = os.path.join(Basic_Args.data_path, 'attr_biomarkers.csv')
            total = sum(1 for line in open(filename))
            if total == len(sample_list)+1:
                not_exists_flag = False
        time.sleep(3)
    time.sleep(3)
    
    test_dataloader = MVI_Dataloader_mvi_test(Basic_Args)

    if Basic_Args.use_attr:
        n_input_channels = 6
    else:
        n_input_channels = 2

    model = generate_model(model_depth=10, n_input_channels=n_input_channels, n_classes=2, args=Basic_Args)
    model.load(Basic_Args.model_load_path)
    model.cuda()
    model.eval()

    cap1_list, cap2_list, cap3_list = None, None, None
    fen1_list, fen2_list, fen3_list = None, None, None
    num_list, radio_list, have_ace_or_not_list = None, None, None

    c = open(os.path.join(Basic_Args.data_path, 'final_predict.csv'),"w")
    writer = csv.writer(c)
    head = ['case_id', 'cap1', 'cap2', 'cap3', 'fen1', 'fen2', 'fen3', 'tmv_num', 'tmv_radio', 'ace', 'max_side', 'resize_factor', 'mvi_score']
    writer.writerow(head)
    new_each_line_list = []

    epoch_finish_flag = False
    with torch.no_grad():
        for k in range(100000):
            if epoch_finish_flag:
                break
            data_dict = next(test_dataloader)

            all_data = data_dict['all_data']
            resize_factor = data_dict['resize_factor']
            name_list = data_dict['name_list'].tolist()
            sample_finish_flag = data_dict['sample_finish_flag']

            if Basic_Args.use_bio_marker:
                cap1_list = data_dict['cap1_list']
                cap2_list = data_dict['cap2_list']

                fen1_list = data_dict['fen1_list']
                fen2_list = data_dict['fen2_list']

                num_list = data_dict['num_list']
                radio_list = data_dict['radio_list']
                have_ace_or_not_list = data_dict['have_ace_or_not_list']

            all_data = maybe_to_torch(all_data)
            if torch.cuda.is_available():
                all_data = to_cuda(all_data)

            score = model(all_data, resize_factor, cap1_list, cap2_list, cap3_list, fen1_list, fen2_list, fen3_list, num_list, radio_list, have_ace_or_not_list)
            score_softmax = F.softmax(score, dim=1)
            pred = score_softmax.data.max(1)[1]
            score_list = score_softmax[:,1]
            # print(name_list, score_softmax)

            bio_marker_csv_ori = os.path.join(Basic_Args.data_path, 'attr_biomarkers.csv')
            with open(bio_marker_csv_ori, 'r') as f:
                reader = csv.reader(f)
                for each_line in reader:
                    if each_line[0] in name_list:
                        th = name_list.index(each_line[0])
                        new_each_line = copy.deepcopy(each_line)
                        new_each_line.append(score_list[th].item())
                        new_each_line_list.append(new_each_line)

            if sample_finish_flag:
                epoch_finish_flag = True
            else:
                epoch_finish_flag = False

    for new_each_line in new_each_line_list:
        writer.writerow(new_each_line)

    os.remove(bio_marker_csv_ori)
    sample_list = os.listdir(Basic_Args.data_path)
    sample_list = [os.path.join(Basic_Args.data_path, each_sample) for each_sample in sample_list if 'csv' not in each_sample]
    for each_sample in sample_list:
        shutil.rmtree(each_sample)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='', help='raw_data_path')
    parser.add_argument('--model_load_path', type=str, default='', help='where to save')

    parser.add_argument('--use_attr', action='store_true')
    parser.add_argument('--use_ourtriplet', action='store_true')
    parser.add_argument('--use_bio_marker', action='store_true') # for test
    
    args = parser.parse_args()
    test(args)
    print('mvi predict finish!')
