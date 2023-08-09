import os
import pandas as pd
import numpy as np
import csv
import SimpleITK as sitk
import argparse

# 所有的可用数据拉清单
def all_train_data2txt(args):
    detailed_save_path = os.path.join(args.save_path, 'all_usable_slices.txt')
    all_trainable_slices_list =  open(detailed_save_path, 'w')
    all_sampeles = os.listdir(args.data_path)
    all_sampeles.sort()
    for each_sample in all_sampeles:
        each_sample_path = os.path.join(args.data_path, each_sample)
        all_slices_of_each_sample = os.listdir(each_sample_path)
        all_slices_of_each_sample.sort()
        for each_slice in all_slices_of_each_sample:
            name = each_slice[:-4]
            all_trainable_slices_list.write(name + '\n')

# 统计label的占比
def get_label_statistics(args):
    statistics_save_path = os.path.join(args.save_path, 'cap_fen_statistics.csv')
    c = open(statistics_save_path,"w")
    writer = csv.writer(c)
    csv_head = ['UID', 'cap+', 'cap++', 'cap+++', 'fen+', 'fen++', 'fen+++']
    writer.writerow(csv_head)
    all_sampeles = os.listdir(args.data_path)
    for each_sample in all_sampeles:
        raw_nii_path = args.data_path.replace('train_data', 'raw_data')
        raw_nii_path = os.path.join(raw_nii_path, each_sample)
        img_path = os.path.join(raw_nii_path, each_sample+'_img_5mm.nii')
        roi_path = os.path.join(raw_nii_path, each_sample+'_roi_5mm.nii')
        cap_path = os.path.join(raw_nii_path, each_sample+'_cap_5mm.nii')
        fen_path = os.path.join(raw_nii_path, each_sample+'_fen_5mm.nii')

        cap_itk = sitk.ReadImage(cap_path)
        cap_npy = sitk.GetArrayFromImage(cap_itk)
        fen_itk = sitk.ReadImage(fen_path)
        fen_npy = sitk.GetArrayFromImage(fen_itk)

        cap1_num = np.sum(cap_npy == 1)
        cap2_num = np.sum(cap_npy == 2)
        cap3_num = np.sum(cap_npy == 3)
        cap1_score = cap1_num/(cap1_num+cap2_num+cap3_num)
        cap2_score = cap2_num/(cap1_num+cap2_num+cap3_num)
        cap3_score = cap3_num/(cap1_num+cap2_num+cap3_num)

        fen1_num = np.sum(fen_npy == 1)
        fen2_num = np.sum(fen_npy == 2)
        fen3_num = np.sum(fen_npy == 3)
        fen1_score = fen1_num/(fen1_num+fen2_num+fen3_num)
        fen2_score = fen2_num/(fen1_num+fen2_num+fen3_num)
        fen3_score = fen3_num/(fen1_num+fen2_num+fen3_num)
        statistics = [each_sample, cap1_score, cap2_score, cap3_score, fen1_score, fen2_score, fen3_score]
        writer.writerow(statistics)

# 【注意：属性平衡不一定对应mvi诊断的平衡】
def make_rebalance_sampling(args):
    detailed_save_path = os.path.join(args.save_path, 'all_usable_slices.txt')
    list_train_slices = [a.strip('\n') for a in open(detailed_save_path).readlines()]
    list_train_slices.sort()
    statistics_save_path = os.path.join(args.save_path, 'cap_fen_statistics.csv')

    split_train_path = os.path.join(args.save_path, args.balance_type + '_split_train_slices.txt')
    split_test_path = os.path.join(args.save_path, args.balance_type + '_split_test_slices.txt')

    train_uid_list = []
    for name in list_train_slices:
        if name[0:6] not in train_uid_list:
            train_uid_list.append(name[0:6]) # uid为6位
    # print(train_uid_list)

    data = pd.read_csv(statistics_save_path, encoding="gb2312")
    UID = data['UID'].values.tolist()
    train_num = int(len(UID) * args.split_prop)
    test_num = len(UID) - train_num
    UID = np.array(UID)
    cap_list = []

    if args.balance_type == 'cap':
        cap_type_list = ['cap+', 'cap++', 'cap+++']
    elif args.balance_type == 'fen':
        cap_type_list = ['fen+', 'fen++', 'fen+++']

    for cap_id in cap_type_list:
        cap_list.append(data[cap_id].values.tolist())  # (3, N)

    cap_list = np.array(cap_list)
    c_argmax = np.argmax(cap_list, axis=0)
    c1_list = UID[c_argmax == 0]
    c2_list = UID[c_argmax == 1]
    c3_list = UID[c_argmax == 2]
    train_c1_list = [a for a in c1_list if a in train_uid_list]
    train_c2_list = [a for a in c2_list if a in train_uid_list]
    train_c3_list = [a for a in c3_list if a in train_uid_list]
    print("train all {} |  c1 {}, c2 {}, c3 {}".format(len(train_uid_list), len(train_c1_list), len(train_c2_list), len(train_c3_list)))
    rebalance = np.round(len(train_c1_list) / len(train_c2_list + train_c3_list))

    capfull_v3_rebalance_train = open(split_train_path, 'w')
    capfull_v3_rebalance_test = open(split_test_path, 'w')


    if len(train_c2_list + train_c3_list) > train_num:
        train_used_uid = (train_c2_list + train_c3_list)[:train_num]
        test_used_uid = (train_c2_list + train_c3_list)[train_num:] + train_c1_list
    else:
        train_used_uid = (train_c2_list + train_c3_list) + train_c1_list[:train_num-len(train_c2_list + train_c3_list)]
        test_used_uid = train_c1_list[train_num-len(train_c2_list + train_c3_list):]


    for name in list_train_slices:
        if name[0:6] in train_used_uid:
            capfull_v3_rebalance_train.write(name+'\n')
        elif name[0:6] in test_used_uid:
            capfull_v3_rebalance_test.write(name+'\n')
        else:
            raise ValueError

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='', help='train_data_path')
    parser.add_argument('--save_path', type=str, default='', help='where to save')
    parser.add_argument('--balance_type', type=str, default='cap', help='use which to balance')
    parser.add_argument('--split_prop', type=float, default=0.8)
    args = parser.parse_args()

    # all_train_data2txt(args)
    # get_label_statistics(args)
    make_rebalance_sampling(args)

