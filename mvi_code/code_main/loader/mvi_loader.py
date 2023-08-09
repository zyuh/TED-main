import csv
import numpy as np
import torch
import random
from collections import OrderedDict
import os
from ARTERY.augmentations.augment_spatial import augment_spatial, augment_spatial_mvi_1, augment_spatial_mvi_2
from ARTERY.augmentations.augment_more import *
from utils.file_ops import *
from skimage.transform import resize
from multiprocessing import Pool
import threading
from threading import Lock,Thread
from scipy.ndimage import gaussian_filter
import torch.backends.cudnn as cudnn
from ARTERY.config.config_mvi_onestream import * 


# 相关系数（绝对值）表示动态margin



def get_range_val(value, rnd_type="uniform"):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 2:
            if value[0] == value[1]:
                n_val = value[0]
            else:
                orig_type = type(value[0])
                if rnd_type == "uniform":
                    n_val = random.uniform(value[0], value[1])
                elif rnd_type == "normal":
                    n_val = random.normalvariate(value[0], value[1])
                n_val = orig_type(n_val)
        elif len(value) == 1:
            n_val = value[0]
        else:
            raise RuntimeError("value must be either a single vlaue or a list/tuple of len 2")
        return n_val
    else:
        return value


def augment_gaussian_noise(v_data, a_data, noise_variance=(0, 0.1), p_per_channel=1):

    if np.random.uniform() <= p_per_channel:
        variance = random.uniform(noise_variance[0], noise_variance[1])
        v_data = v_data + np.random.normal(0.0, variance, size=v_data.shape)

        variance = random.uniform(noise_variance[0], noise_variance[1])
        a_data = a_data + np.random.normal(0.0, variance, size=a_data.shape)

    return v_data, a_data


def augment_gaussian_blur(v_data, a_data, sigma_range=(1, 5), p_per_channel=1):
    
    if np.random.uniform() <= p_per_channel:
        sigma = get_range_val(sigma_range)
        v_data = gaussian_filter(v_data, sigma, order=0)

        sigma = get_range_val(sigma_range)
        a_data = gaussian_filter(a_data, sigma, order=0)

    return v_data, a_data



def augment_brightness_multiplicative(v_data, a_data, multiplier_range=(0.5, 2), p_per_channel=1):
    if np.random.uniform() <= p_per_channel:
        multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
        v_data = v_data * multiplier

        multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
        a_data = a_data * multiplier

    return v_data, a_data



def augment_gamma(data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, retain_stats=True):
    if invert_image:
        data_sample = - data_sample

    if retain_stats:
        mn = data_sample.mean()
        sd = data_sample.std()


    if np.random.random() < 0.5 and gamma_range[0] < 1:
        gamma = np.random.uniform(gamma_range[0], 1)
    else:
        gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
    minm = data_sample.min()
    rnge = data_sample.max() - minm
    data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm


    if retain_stats:
        data_sample = data_sample - data_sample.mean() + mn
        data_sample = data_sample / (data_sample.std() + 1e-8) * sd

    if invert_image:
        data_sample = - data_sample
    return data_sample



def augment_contrast(data_sample, contrast_range=(0.75, 1.25), preserve_range=True):

    mn = data_sample.mean()
    if preserve_range:
        minm = data_sample.min()
        maxm = data_sample.max()


    if np.random.random() < 0.5 and contrast_range[0] < 1:
        factor = np.random.uniform(contrast_range[0], 1)
    else:
        factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])

    data_sample = (data_sample - mn) * factor + mn


    if preserve_range:
        data_sample[data_sample < minm] = minm
        data_sample[data_sample > maxm] = maxm

    return data_sample


def get_bbox_3d(roi): # 根据roi得到bbox
    zmin = np.min(np.nonzero(roi)[0])
    zmax = np.max(np.nonzero(roi)[0])
    rmin = np.min(np.nonzero(roi)[1])
    rmax = np.max(np.nonzero(roi)[1])
    cmin = np.min(np.nonzero(roi)[2])
    cmax = np.max(np.nonzero(roi)[2])
    return zmin, zmax, rmin, rmax, cmin, cmax


def resize_segmentation(segmentation, new_shape, order=3, cval=0):
    '''
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)
        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped


class RandomRotFlip(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = m = np.arange(8).reshape((2,2,2))(image, k)
        label = np.rot90(label, k) # 维度默认(0,1)
        axis = np.random.randint(0, 3)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return {'image': image, 'label': label}


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)],
                             mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}




# 关于如何判断两个样本的属性是否相似，我们设置一种规则将点数百分比数值进行分类
# 9 个样本 + 1 放缩尺寸
def score_hard_transform_rank(attrs_score_list, n_rank=3):
    # ['location', 'size', 'ace', 'cap+', 'cap++', 'fen+', 'fen++', 'radio', 'num']
    # ['size', 'ace', 'cap+', 'cap++', 'fen+', 'fen++', 'radio', 'num']
    assert n_rank==3 or n_rank==5
    # location_score = attrs_score_list[0]
    size_score = attrs_score_list[0] # 5 - 64 - 250 # 取最大边
    # ace
    cap1_score = attrs_score_list[2]
    cap2_score = attrs_score_list[3]
    fen1_score = attrs_score_list[4]
    fen2_score = attrs_score_list[5]
    radio_score = attrs_score_list[6]
    num_score = attrs_score_list[7]

    if n_rank==3:
        if size_score < 35:
            size_rank = 1
        elif size_score < 150:
            size_rank = 2
        else:
            size_rank = 3
        
        if cap1_score < 0.33:
            cap1_rank = 1
        elif cap1_score < 0.67:
            cap1_rank = 2
        else:
            cap1_rank = 3

        if cap2_score < 0.33:
            cap2_rank = 1
        elif cap2_score < 0.67:
            cap2_rank = 2
        else:
            cap2_rank = 3

        if fen1_score < 0.33:
            fen1_rank = 1
        elif fen1_score < 0.67:
            fen1_rank = 2
        else:
            fen1_rank = 3

        if fen2_score < 0.33:
            fen2_rank = 1
        elif fen2_score < 0.67:
            fen2_rank = 2
        else:
            fen2_rank = 3

        if radio_score < 0.04:  # [0,0.079,0.41]
            radio_rank = 1
        elif radio_score < 0.2:
            radio_rank = 2
        else:
            radio_rank = 3

        if num_score < 230:
            num_rank = 1
        elif num_score < 1600:
            num_rank = 2
        else:
            num_rank = 3
    
    else:
        if size_score < 25:  # 5 - 64 - 250
            size_rank = 1
        elif size_score < 60:
            size_rank = 2
        elif size_score < 90:
            size_rank = 3
        elif size_score < 160:
            size_rank = 4
        else:
            size_rank = 5
        
        if cap1_score < 0.15: # mean == 0.27
            cap1_rank = 1
        elif cap1_score < 0.3:
            cap1_rank = 2
        elif cap1_score < 0.5:
            cap1_rank = 3
        elif cap1_score < 0.7:
            cap1_rank = 4
        else:
            cap1_rank = 5

        if cap2_score < 0.15:
            cap2_rank = 1
        elif cap2_score < 0.3:
            cap2_rank = 2
        elif cap2_score < 0.5:
            cap2_rank = 3
        elif cap2_score < 0.7:
            cap2_rank = 4
        else:
            cap2_rank = 5

        if fen1_score < 0.15:
            fen1_rank = 1
        elif fen1_score < 0.3:
            fen1_rank = 2
        elif fen1_score < 0.5:
            fen1_rank = 3
        elif fen1_score < 0.7:
            fen1_rank = 4
        else:
            fen1_rank = 5

        if fen2_score < 0.15:
            fen2_rank = 1
        elif fen2_score < 0.3:
            fen2_rank = 2
        elif fen2_score < 0.5:
            fen2_rank = 3
        elif fen2_score < 0.7:
            fen2_rank = 4
        else:
            fen2_rank = 5

        if radio_score < 0.04:  # [0,0.079,0.41]
            radio_rank = 1
        elif radio_score < 0.1:
            radio_rank = 2
        elif radio_score < 0.15:
            radio_rank = 3
        elif radio_score < 0.3:
            radio_rank = 4
        else:
            radio_rank = 5

        if num_score < 230:
            num_rank = 1
        elif num_score < 800:
            num_rank = 2
        elif num_score < 2000:
            num_rank = 3
        elif num_score < 3200:
            num_rank = 4
        else:
            num_rank = 5


    attrs_ranks_list = [size_rank, attrs_score_list[1], cap1_rank, cap2_rank, fen1_rank, fen2_rank, radio_rank, num_rank]

    return attrs_ranks_list


def get_difference_hard(attrs_list1, attrs_list2, correlation_coeff, n_rank=3, alpha=1):
    # 按离散值计算样本之间的属性差异： online比较每两个样本属性分数之间的差异（绝对值加权和）
    # ['location', 'size', 'ace', 'cap+', 'cap++', 'fen+', 'fen++', 'radio', 'num']
    difference_hard_list = torch.zeros([len(attrs_list1)])

    # 量化除 location(center point) 以外的属性
    attrs_ranks_list1 = score_hard_transform_rank(attrs_list1, n_rank)
    attrs_ranks_list2 = score_hard_transform_rank(attrs_list2, n_rank)

    # for location (center point)
    # difference_score_location = torch.abs(torch.tensor(attrs_list1[0]) - torch.tensor(attrs_list2[0]))
    # if n_rank == 3:
    #     if torch.max(difference_score_location) < 30: # 考虑偏差最大的方向
    #         difference_rank_location = 1
    #     elif torch.max(difference_score_location) < 70:
    #         difference_rank_location = 2
    #     else:
    #         difference_rank_location = 3
    # else:
    #     if torch.max(difference_score_location) < 20:
    #         difference_rank_location = 1
    #     elif torch.max(difference_score_location) < 50:
    #         difference_rank_location = 2
    #     elif torch.max(difference_score_location) < 80:
    #         difference_rank_location = 3
    #     elif torch.max(difference_score_location) < 120:
    #         difference_rank_location = 4
    #     else:
    #         difference_rank_location = 5

    # difference_hard_list[0] = difference_rank_location 
    difference_hard_list = torch.abs(torch.tensor(attrs_ranks_list1) - torch.tensor(attrs_ranks_list2))

    difference_hard_list_weighted = torch.sum(difference_hard_list * correlation_coeff)

    return difference_hard_list_weighted * alpha


def get_difference_soft(attrs_list1, attrs_list2, correlation_coeff, n_rank=3, alpha=1):
    # 按连续值计算样本之间的属性差异： online比较每两个样本属性分数之间的差异（绝对值加权和）

    difference_soft_list = torch.abs(torch.tensor(attrs_list1) - torch.tensor(attrs_list2)) #l1
    # difference_soft_list = torch.sqrt(torch.sum(torch.square(torch.tensor(attrs_list1)-torch.tensor(attrs_list2)))) #l2

    difference_soft_list_weighted = torch.sum(difference_soft_list * correlation_coeff)
    # difference_soft_list_weighted = torch.cosine_similarity(torch.tensor(attrs_list1), torch.tensor(attrs_list2), dim=0) #cos
    return difference_soft_list_weighted * alpha



class MVI_Dataloader_mvi(object):
    def __init__(self, Basic_Args, phase, balance=False, logger=None):
        self.args = Basic_Args
        self.phase = phase
        self.balance = balance
        self.logger = logger

        self.patch_size = self.args.patch_size
        self.batch_size = self.args.batch_size
        self.mode = self.args.mode  # 属性的mode 选择预测的还是gt的

        if self.args.valid_plus_test and self.phase != 'train':
            self.lines, self.line_head = self.prepare_dataV5(split_flag=self.phase, normalization_flag=self.args.normalization_flag)
        else:
            self.lines, self.line_head = self.prepare_dataV4(split_flag=self.phase, normalization_flag=self.args.normalization_flag)

        self._lines = copy.deepcopy(self.lines)

        self.use_attribute = self.args.use_attribute
        self.use_bio_marker = self.args.use_bio_marker
        self.do_rotation = self.args.do_rotation

        if self.args.triplet_flag:
            # step1: 得到不同属性的相关系数, 其中size取最大边
            '''location 相关性太小不要了'''
            # self.metrics_attrs = ['location', 'size', 'ace', 'cap+', 'cap++', 'fen+', 'fen++', 'radio', 'num']
            # self.correlation_coeff = [0.025,   0.5,    0.21,  0.40,   0.33,   0.12,   0.13,     0.25,   0.30] # CODE/myxgboost.py
            self.metrics_attrs = ['size', 'ace', 'cap+', 'cap++', 'fen+', 'fen++', 'radio', 'num']
            self.correlation_coeff = [0.5,    0.21,  0.40,   0.33,   0.12,   0.13,     0.25,   0.30] # CODE/myxgboost.py
            # self.correlation_coeff = [1.0,    1.0,    1.0,   1.0,    1.0,     1.0,     1.0,     1.0]
            # self.correlation_coeff = [1.5,    1.21,    1.40,   1.33,    1.12,     1.13,     1.25,     1.30]
            self.correlation_coeff = torch.tensor(self.correlation_coeff)

            # margin 默认 0.3 # location 始终是soft的
            if self.args.triplet_hard:
                self.attrs_to_difference_fn = get_difference_hard
            else:
                self.attrs_to_difference_fn = get_difference_soft

            # step2: 得到属性分级的规则，判断两者属性是否相似
            # 考虑的属性有 location(中心点位置在一定范围内32？64？) size(肿瘤resize前的尺寸10-64-224)  ace(0/1)
            # cap+   cap++   fen+   fen++    radio   num  (均值附近, 最大，最小) # 分成三档 （硬性划分，提前规定好）
        cap1_list, cap2_list, cap3_list, fen1_list, fen2_list, fen3_list,\
            radio_list, num_list, crop_size_plus_list, ace_list = self.get_mean_std_for_each_attr(mode=self.mode)
        self.cap1_mean, self.cap1_std = np.mean(cap1_list), np.std(cap1_list)
        self.cap2_mean, self.cap2_std = np.mean(cap2_list), np.std(cap2_list)
        self.cap3_mean, self.cap3_std = np.mean(cap3_list), np.std(cap3_list)
        self.fen1_mean, self.fen1_std = np.mean(fen1_list), np.std(fen1_list)
        self.fen2_mean, self.fen2_std = np.mean(fen2_list), np.std(fen2_list)
        self.fen3_mean, self.fen3_std = np.mean(fen3_list), np.std(fen3_list)

        self.cap1_max, self.cap1_min = np.max(cap1_list), np.min(cap1_list)
        self.cap2_max, self.cap2_min = np.max(cap2_list), np.min(cap2_list)
        self.cap3_max, self.cap3_min = np.max(cap3_list), np.min(cap3_list)
        self.fen1_max, self.fen1_min = np.max(fen1_list), np.min(fen1_list)
        self.fen2_max, self.fen2_min = np.max(fen2_list), np.min(fen2_list)
        self.fen3_max, self.fen3_min = np.max(fen3_list), np.min(fen3_list)

        self.radio_mean, self.radio_std = np.mean(radio_list), np.std(radio_list)
        self.num_mean, self.num_std = np.mean(num_list), np.std(num_list)
        self.crop_size_mean, self.crop_size_std = np.mean(crop_size_plus_list), np.std(crop_size_plus_list)
        self.ace_mean, self.ace_std = np.mean(ace_list), np.std(ace_list)

        self.radio_max, self.radio_min = np.max(radio_list), np.min(radio_list)
        self.num_max, self.num_min = np.max(num_list), np.min(num_list)
        self.crop_size_max, self.crop_size_min = np.max(crop_size_plus_list), np.min(crop_size_plus_list)
        self.ace_max, self.ace_min = np.max(ace_list), np.min(ace_list)


        if self.phase!='train':
            self.do_rotation = False

        self.data_shape, self.data_shape_tmp, self.each_data_shape = self.determine_shapes()


    def get_mean_std_for_each_attr(self, mode='our_pred'):
        assert self.mode == 'our_pred' or self.mode == 'our_gt' or self.mode == 'ori'

        pred_root_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/preprocessed_data/nnUNetData_plans_v2.1_stage1/resized_data_pred'
        pred_root_path_train = join(pred_root_path, 'train')
        pred_train_list = subfiles(pred_root_path_train, join=True, suffix='.npy')

        pred_root_path_valid = join(pred_root_path, 'valid')
        pred_valid_list = subfiles(pred_root_path_valid, join=True, suffix='.npy')

        pred_root_path_test = join(pred_root_path, 'test')
        pred_test_list = subfiles(pred_root_path_test, join=True, suffix='.npy')

        pred_list = pred_train_list+ pred_valid_list + pred_test_list
        pred_list.sort()

        cap1_list, cap2_list, cap3_list = [],[],[]
        fen1_list, fen2_list, fen3_list = [],[],[]
        # ace 不需要
        radio_list, num_list = [],[]
        resize_factor_list = []
        center_point_list = []
        crop_size_plus_list = []

        ace_list = []


        for th in range(len(pred_list)):
            each_pred_path = pred_list[th]
            each_gt_path = each_pred_path.replace('resized_data_pred', 'resized_data_gt')
            each_pkl_path = each_gt_path.replace('.npy', '.pkl')
            each_pkl = load_pickle(each_pkl_path)

            resize_factor = each_pkl['resize_factor']
            # center_point = each_pkl['ori_center_point']  # location 不考虑了
            crop_size_plus = each_pkl['ori_rect_size']

            resize_factor_list.append(float(resize_factor))
            # center_point_list.append(center_point)
            crop_size_plus_list.append(np.max(crop_size_plus))
 
            only_id = each_pkl_path.split('/')[-1][:-4]

            bio_marker_csv = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/preprocessed_data/all_pred_biomarker.csv'

            with open(bio_marker_csv, 'r') as f:
                reader = csv.reader(f)
                for each_line in reader:
                    if each_line[0] == 'UID': # 跳过表头
                        continue
                    if only_id == each_line[1]:
                        if  mode=='our_pred':
                            cap1_list.append(float(each_line[27]))
                            cap2_list.append(float(each_line[28]))
                            cap3_list.append(float(each_line[29]))
                            fen1_list.append(float(each_line[18]))
                            fen2_list.append(float(each_line[19]))
                            fen3_list.append(float(each_line[20]))

                            if each_line[7] != 'None':
                                radio_list.append(float(each_line[7]))
                            else:
                                radio_list.append(0)
                            ace_list.append(float(each_line[8]))
                            if each_line[6]!='None':
                                num_list.append(int(each_line[6]))
                            else:
                                num_list.append(0)
                        elif mode=='ori':
                            cap1_list.append(float(each_line[9]))
                            cap2_list.append(float(each_line[10]))
                            cap3_list.append(float(each_line[11]))
                            fen1_list.append(float(each_line[12]))
                            fen2_list.append(float(each_line[13]))
                            fen3_list.append(float(each_line[14]))

                            if each_line[5] != 'None':
                                radio_list.append(float(each_line[5]))
                            else:
                                radio_list.append(0)
                            ace_list.append(float(each_line[8]))

                            if each_line[4]!='None':
                                num_list.append(int(each_line[4]))
                            else:
                                num_list.append(0)
                        elif mode=='our_gt':
                            cap1_list.append(float(each_line[15]))
                            cap2_list.append(float(each_line[16]))
                            cap3_list.append(float(each_line[17]))
                            fen1_list.append(float(each_line[18]))
                            fen2_list.append(float(each_line[19]))
                            fen3_list.append(float(each_line[20]))

                            if each_line[5] != 'None':
                                radio_list.append(float(each_line[5]))
                            else:
                                radio_list.append(0)
                            ace_list.append(float(each_line[8]))

                            if each_line[4]!='None':
                                num_list.append(int(each_line[4]))
                            else:
                                num_list.append(0)

            f.close()

        return cap1_list, cap2_list, cap3_list, fen1_list, fen2_list, fen3_list,\
                radio_list, num_list, crop_size_plus_list, ace_list


    def prepare_dataV4(self, split_flag='train', normalization_flag='roi_cap'): # for train  valid  test
        pred_root_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/preprocessed_data/nnUNetData_plans_v2.1_stage1/resized_data_pred'
        pred_root_path = join(pred_root_path, split_flag)
        pred_list = subfiles(pred_root_path, join=True, suffix='.npy')
        
        if split_flag=='train':
            assert len(pred_list) == 324
        elif split_flag=='valid':
            assert len(pred_list) == 49
        elif split_flag=='test':
            assert len(pred_list) == 93
        else:
            raise ValueError
        pred_list.sort()

        positive_list = []
        negative_list = []

        num_gt_list = []
        num_pred_list = []

        for th in range(len(pred_list)):
            each_pred_path = pred_list[th]
            each_gt_path = each_pred_path.replace('resized_data_pred', 'resized_data_gt')
            each_pkl_path = each_gt_path.replace('.npy', '.pkl')
            each_pkl = load_pickle(each_pkl_path)

            resize_factor = each_pkl['resize_factor']
            mvi = each_pkl['mvi']
            have_ace = each_pkl['have_ace']
            have_artery = each_pkl['have_artery']
            only_id = each_pkl_path.split('/')[-1][:-4]

            center_point = each_pkl['ori_center_point'] #都是list
            crop_size_plus = each_pkl['ori_rect_size']# 更能反应尺寸

            each_patient_data_list = [each_gt_path, each_pred_path, each_pkl_path, resize_factor, mvi, have_ace, have_artery]
            bio_marker_csv = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/preprocessed_data/all_pred_biomarker.csv'
            bio_marker = []
            with open(bio_marker_csv, 'r') as f:
                reader = csv.reader(f)
                for each_line in reader:
                    if each_line[0] == 'UID': # 跳过表头
                        line_head = copy.deepcopy(each_line)
                        head_plus = ['each_gt_path', 'each_pred_path', 'each_pkl_path', 'resize_factor', 'mvi', 'have_ace', 'have_artery']
                        line_head = head_plus + line_head + ['center_point', 'crop_size_plus'] #其实是原始尺寸
                        continue
                    if only_id == each_line[1]:
                        bio_marker = copy.deepcopy(each_line)

                        if each_line[4]!='None': # 用来统计的
                            num_gt_list.append(int(each_line[4]))
                        if each_line[6]!='None':
                            num_pred_list.append(int(each_line[6]))
                        break
            f.close()
            each_patient_data_list = each_patient_data_list + bio_marker 
            each_patient_data_list.append(center_point)
            each_patient_data_list.append(crop_size_plus)
            assert len(line_head) == len(each_patient_data_list)

            mvi = int(mvi)
            assert mvi==0 or mvi==1
            if mvi == 1:
                positive_list.append(each_patient_data_list)
            elif mvi==0:
                negative_list.append(each_patient_data_list)
            else:
                raise ValueError

        self.gt_mean_num = np.mean(num_gt_list) 
        self.gt_std_num = np.std(num_gt_list)
        self.pred_mean_num = np.mean(num_pred_list)
        self.pred_std_num = np.std(num_pred_list)

        lists = positive_list + negative_list
        self.logger.log('before {} balance, prepare {} data where positive {} and negative {}'.format(self.args.data_balance, len(lists), len(positive_list), len(negative_list)))
        
        self.positive_list = positive_list
        self.negative_list = negative_list

        if self.balance: # only for train
            if self.args.data_balance == 'upsample':
                self.scale = [1, len(negative_list) // len(positive_list)]
                positive_list = positive_list * (len(negative_list) // len(positive_list))  # 增加正样本
            elif self.args.data_balance == 'downsample':
                self.scale = [len(negative_list) // len(positive_list), 1]
                negative_list = random.sample(negative_list, 2*len(positive_list))  # 减少负样本
            else:
                raise ValueError
            lists = positive_list + negative_list
            self.logger.log('after {} balance, prepare {} data where positive {} and negative {} and scale {}'.format(self.args.data_balance, len(lists), len(positive_list), len(negative_list), self.scale))

        if self.phase == 'train':
            self.logger.log('for example, {}'.format(lists[0]))
        return lists, line_head


    def prepare_dataV5(self, split_flag='valid', normalization_flag='roi_cap'): # phase != 'train' | 将valid和test合并
        assert split_flag != 'train'
        pred_root_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/preprocessed_data/nnUNetData_plans_v2.1_stage1/resized_data_pred'
        pred_root_path_valid = join(pred_root_path, 'valid')
        pred_valid_list = subfiles(pred_root_path_valid, join=True, suffix='.npy')

        pred_root_path_test = join(pred_root_path, 'test')
        pred_test_list = subfiles(pred_root_path_test, join=True, suffix='.npy')

        pred_list = pred_valid_list + pred_test_list
        
        assert len(pred_list) == 142

        pred_list.sort()

        positive_list = []
        negative_list = []

        num_gt_list = []
        num_pred_list = []

        for th in range(len(pred_list)):
            each_pred_path = pred_list[th]
            each_gt_path = each_pred_path.replace('resized_data_pred', 'resized_data_gt')
            each_pkl_path = each_gt_path.replace('.npy', '.pkl')
            each_pkl = load_pickle(each_pkl_path)

            resize_factor = each_pkl['resize_factor']
            mvi = each_pkl['mvi']
            have_ace = each_pkl['have_ace']
            have_artery = each_pkl['have_artery']
            only_id = each_pkl_path.split('/')[-1][:-4]

            center_point = each_pkl['ori_center_point'] #都是list
            crop_size_plus = each_pkl['ori_rect_size']# 更能反应尺寸

            each_patient_data_list = [each_gt_path, each_pred_path, each_pkl_path, resize_factor, mvi, have_ace, have_artery]
            bio_marker_csv = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/preprocessed_data/all_pred_biomarker.csv'
            bio_marker = []
            with open(bio_marker_csv, 'r') as f:
                reader = csv.reader(f)
                for each_line in reader:
                    if each_line[0] == 'UID': # 跳过表头
                        line_head = copy.deepcopy(each_line)
                        head_plus = ['each_gt_path', 'each_pred_path', 'each_pkl_path', 'resize_factor', 'mvi', 'have_ace', 'have_artery']
                        line_head = head_plus + line_head + ['center_point', 'crop_size_plus'] #其实是原始尺寸
                        continue
                    if only_id == each_line[1]:
                        bio_marker = copy.deepcopy(each_line)
                        if each_line[4]!='None':
                            num_gt_list.append(int(each_line[4]))
                        if each_line[6]!='None':
                            num_pred_list.append(int(each_line[6]))
                        break
            f.close()
            each_patient_data_list = each_patient_data_list + bio_marker
            each_patient_data_list.append(center_point)
            each_patient_data_list.append(crop_size_plus)
            assert len(line_head) == len(each_patient_data_list)

            mvi = int(mvi)
            assert mvi==0 or mvi==1
            if mvi == 1:
                positive_list.append(each_patient_data_list)
            elif mvi==0:
                negative_list.append(each_patient_data_list)
            else:
                raise ValueError

        # print(num_gt_list)
        self.gt_mean_num = np.mean(num_gt_list) 
        self.gt_std_num = np.std(num_gt_list)
        self.pred_mean_num = np.mean(num_pred_list)
        self.pred_std_num = np.std(num_pred_list)

        lists = positive_list + negative_list
        self.logger.log('before {} balance, prepare {} data where positive {} and negative {}'.format(self.args.data_balance, len(lists), len(positive_list), len(negative_list)))

        return lists, line_head


    def determine_shapes(self):
        if self.use_attribute: #one_stream
            num_color_channels = 6
        else:
            num_color_channels = 2

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        data_shape_tmp = (len(self.lines) % self.batch_size, num_color_channels, *self.patch_size)
        each_data_shape = (1, num_color_channels, *self.patch_size)

        return data_shape, data_shape_tmp, each_data_shape


    def generate_train_batch_V3(self):
        selected_lines = []
        if len(self.lines) >= self.batch_size:
            line_selected_th_list = np.random.choice(len(self.lines), self.batch_size, False, None)
            all_data = np.zeros(self.data_shape, dtype=np.float32)
        else:
            line_selected_th_list = np.random.choice(len(self.lines), len(self.lines), False, None)
            all_data = np.zeros(self.data_shape_tmp, dtype=np.float32)


        for line_selected_th in line_selected_th_list:
            selected_lines.append(self.lines[line_selected_th])

        for remove_line in selected_lines:
            self.lines.remove(remove_line)

        lines_plus = copy.deepcopy(selected_lines)

        if len(self.lines) == 0:
            sample_finish_flag = True
            self.lines = copy.deepcopy(self._lines)
            random.shuffle(self.lines)
        else:
            sample_finish_flag = False

        # ================================================ #
        if self.phase == 'train':
            selected_lines = []
            line_selected_th_list = np.random.choice(len(self.positive_list), self.batch_size//2, False, None) # 都是balance之前的
            for line_selected_th in line_selected_th_list:
                selected_lines.append(self.positive_list[line_selected_th])

            line_selected_th_list = np.random.choice(len(self.negative_list), self.batch_size//2, False, None)
            for line_selected_th in line_selected_th_list:
                selected_lines.append(self.negative_list[line_selected_th])

            all_data = np.zeros(self.data_shape, dtype=np.float32)
            lines_plus = copy.deepcopy(selected_lines)
            random.shuffle(lines_plus)
        # ================================================ #
        # epoch 遍历的结果不一定好，不使用balance 直接以interation采样
        # if self.phase == 'train':
        #     selected_lines = []
        #     line_selected_th_list = np.random.choice(len(self._lines), self.batch_size, False, None)
        #     for line_selected_th in line_selected_th_list:
        #         selected_lines.append(self._lines[line_selected_th])
        #     all_data = np.zeros(self.data_shape, dtype=np.float32)
        #     lines_plus = copy.deepcopy(selected_lines)
        # ================================================ #
        magin_array = torch.zeros([len(lines_plus), len(lines_plus), len(lines_plus)])
        attrs_feature_list = []

        label_list = [] #mvi
        resize_factor_list = []
        name_list = []

        center_point_list = []
        crop_size_plus_list = []

        cap1_list, cap2_list, cap3_list = [], [], []
        fen1_list, fen2_list, fen3_list = [], [], []
        num_list = []
        radio_list = []
        have_ace_or_not_list = []

        for j, line in enumerate(lines_plus):

            all_gt_data = np.load(line[0], 'r')
            all_pred_data = np.load(line[1], 'r')

            resize_factor = float(line[3])
            mvi = int(line[4])
            have_ace = int(line[5])
            have_artery = int(line[6])

            center_point = line[-2]
            crop_size_plus = line[-1] #其实是原始尺寸

            have_ace_or_not = line[self.line_head.index('have_ace_or_not')]
            have_ace_or_not_list.append((int(have_ace_or_not) - self.ace_min) / (self.ace_max - self.ace_min))


            if self.mode == 'ori':
                ori_cap1 = line[self.line_head.index('ori_cap+')]
                ori_cap2 = line[self.line_head.index('ori_cap++')]
                ori_cap3 = line[self.line_head.index('ori_cap+++')]
                ori_fen1 = line[self.line_head.index('ori_fen+')]
                ori_fen2 = line[self.line_head.index('ori_fen++')]
                ori_fen3 = line[self.line_head.index('ori_fen+++')]
                ori_num = line[self.line_head.index('ori_num')]
                ori_radio = line[self.line_head.index('ori_radio')]
                if ori_num == 'None':
                    ori_num = 0
                else:
                    ori_num = float(ori_num)

                if ori_radio == 'None':
                    ori_radio = 0
                else:
                    ori_radio = float(ori_radio)

                # cap1_list.append(float(ori_cap1))
                # cap2_list.append(float(ori_cap2))
                # cap3_list.append(float(ori_cap3))
                # fen1_list.append(float(ori_fen1))
                # fen2_list.append(float(ori_fen2))
                # fen3_list.append(float(ori_fen3))
                # num_list.append(float(ori_num))
                # radio_list.append(float(ori_radio))

                cap1_list.append((float(ori_cap1) - self.cap1_min) / (self.cap1_max - self.cap1_min))
                cap2_list.append((float(ori_cap2) - self.cap2_min) / (self.cap2_max - self.cap2_min))
                cap3_list.append((float(ori_cap3) - self.cap3_min) / (self.cap3_max - self.cap3_min))
                fen1_list.append((float(ori_fen1) - self.fen1_min) / (self.fen1_max - self.fen1_min))
                fen2_list.append((float(ori_fen2) - self.fen2_min) / (self.fen2_max - self.fen2_min))
                fen3_list.append((float(ori_fen3) - self.fen3_min) / (self.fen3_max - self.fen3_min))
                num_list.append((float(ori_num) - self.num_min) / (self.num_max - self.num_min))
                radio_list.append((float(ori_radio) - self.radio_min) / (self.radio_max - self.radio_min))

            elif self.mode == 'our_gt':
                our_gt_cap1 = line[self.line_head.index('our_gt_cap+')]
                our_gt_cap2 = line[self.line_head.index('our_gt_cap++')]
                our_gt_cap3 = line[self.line_head.index('our_gt_cap+++')]
                our_gt_fen1 = line[self.line_head.index('our_gt_fen+')]
                our_gt_fen2 = line[self.line_head.index('our_gt_fen++')]
                our_gt_fen3 = line[self.line_head.index('our_gt_fen+++')]
                ori_num = line[self.line_head.index('ori_num')]
                ori_radio = line[self.line_head.index('ori_radio')]
                if ori_num == 'None':
                    ori_num = 0
                else:
                    ori_num = float(ori_num)
                if ori_radio == 'None':
                    ori_radio = 0
                else:
                    ori_radio = float(ori_radio)

                # cap1_list.append(float(our_gt_cap1))
                # cap2_list.append(float(our_gt_cap2))
                # cap3_list.append(float(our_gt_cap3))
                # fen1_list.append(float(our_gt_fen1))
                # fen2_list.append(float(our_gt_fen2))
                # fen3_list.append(float(our_gt_fen3))
                # num_list.append(float(ori_num))
                # radio_list.append(float(ori_radio))

                cap1_list.append((float(our_gt_cap1) - self.cap1_min) / (self.cap1_max - self.cap1_min))
                cap2_list.append((float(our_gt_cap2) - self.cap2_min) / (self.cap2_max - self.cap2_min))
                cap3_list.append((float(our_gt_cap3) - self.cap3_min) / (self.cap3_max - self.cap3_min))
                fen1_list.append((float(our_gt_fen1) - self.fen1_min) / (self.fen1_max - self.fen1_min))
                fen2_list.append((float(our_gt_fen2) - self.fen2_min) / (self.fen2_max - self.fen2_min))
                fen3_list.append((float(our_gt_fen3) - self.fen3_min) / (self.fen3_max - self.fen3_min))
                num_list.append((float(ori_num) - self.num_min) / (self.num_max - self.num_min))
                radio_list.append((float(ori_radio) - self.radio_min) / (self.radio_max - self.radio_min))

            elif self.mode == 'our_semantic':
                our_semantic_cap1 = line[self.line_head.index('our_semantic_cap+')]
                our_semantic_cap2 = line[self.line_head.index('our_semantic_cap++')]
                our_semantic_cap3 = line[self.line_head.index('our_semantic_cap+++')]
                our_semantic_fen1 = line[self.line_head.index('our_semantic_fen+')]
                our_semantic_fen2 = line[self.line_head.index('our_semantic_fen++')]
                our_semantic_fen3 = line[self.line_head.index('our_semantic_fen+++')]
                ori_num = line[self.line_head.index('ori_num')]
                ori_radio = line[self.line_head.index('ori_radio')]
                if ori_num == 'None':
                    ori_num = 0
                else:
                    ori_num = (float(ori_num)-self.gt_mean_num)/self.gt_std_num 
                if ori_radio == 'None':
                    ori_radio = 0

                cap1_list.append(float(our_semantic_cap1))
                cap2_list.append(float(our_semantic_cap2))
                cap3_list.append(float(our_semantic_cap3))
                fen1_list.append(float(our_semantic_fen1))
                fen2_list.append(float(our_semantic_fen2))
                fen3_list.append(float(our_semantic_fen3))
                num_list.append(float(ori_num))
                radio_list.append(float(ori_radio))

            elif self.mode == 'our_pred':
                pred_cap1 = line[self.line_head.index('pred_cap+')]
                pred_cap2 = line[self.line_head.index('pred_cap++')]
                pred_cap3 = line[self.line_head.index('pred_cap+++')]
                our_gt_fen1 = line[self.line_head.index('our_gt_fen+')]
                our_gt_fen2 = line[self.line_head.index('our_gt_fen++')]
                our_gt_fen3 = line[self.line_head.index('our_gt_fen+++')]
                pred_num = line[self.line_head.index('pred_num')]
                pred_radio = line[self.line_head.index('pred_radio')]
                if pred_num == 'None':
                    pred_num = 0
                else:
                    pred_num = float(pred_num)
                    
                if pred_radio == 'None':
                    pred_radio = 0
                else:
                    pred_radio = float(pred_radio)

                cap1_list.append((float(pred_cap1) - self.cap1_min) / (self.cap1_max - self.cap1_min))
                cap2_list.append((float(pred_cap2) - self.cap2_min) / (self.cap2_max - self.cap2_min))
                cap3_list.append((float(pred_cap3) - self.cap3_min) / (self.cap3_max - self.cap3_min))
                fen1_list.append((float(our_gt_fen1) - self.fen1_min) / (self.fen1_max - self.fen1_min))
                fen2_list.append((float(our_gt_fen2) - self.fen2_min) / (self.fen2_max - self.fen2_min))
                fen3_list.append((float(our_gt_fen3) - self.fen3_min) / (self.fen3_max - self.fen3_min))
                num_list.append((float(pred_num) - self.num_min) / (self.num_max - self.num_min))
                radio_list.append((float(pred_radio) - self.radio_min) / (self.radio_max - self.radio_min))

            else:
                # raise ValueError
                pass

            if self.phase == 'train':
                x = np.random.randint(0, 9)
                y = np.random.randint(0, 9)
                z = np.random.randint(0, 9)
                all_gt_data = all_gt_data[:, :, x:x+64, y:y+64, z:z+64]
                all_pred_data = all_pred_data[:, :, x:x+64, y:y+64, z:z+64]
            else:
                all_gt_data = all_gt_data[:, :, 4:68, 4:68, 4:68]
                all_pred_data = all_pred_data[:, :, 4:68, 4:68, 4:68]


            if self.use_attribute:
                # all_gt_data
                # each_data[0, 0:1] = data_after_resized
                # each_data[0, 1:2] = artery_after_resized
                # each_data[0, 2:3] = ace_after_resized
                # each_data[0, 3:4] = roi_after_resized
                # each_data[0, 4:5] = venous_data_after_resized
                # each_data[0, 5:6] = cap_after_resized
                # each_data[0, 6:7] = fen_after_resized
                artery_data = all_gt_data[0, 0]
                venous_data = all_gt_data[0, 4]
                 
                ace = all_gt_data[0, 2]

                assert len(artery_data.shape) == 3

                if self.mode == 'ori':
                    # all_pred_data
                    # each_data[0, 0:1] = label_cap_after_resized
                    # each_data[0, 1:2] = label_cap_semantic_after_resized
                    # each_data[0, 2:3] = label_fen_after_resized
                    # each_data[0, 3:4] = label_fen_semantic_after_resized
                    # each_data[0, 4:5] = pred_cap_after_resized
                    # each_data[0, 5:6] = new_artery_after_resized
                    # each_data[0, 6:7] = old_artery_after_resized
                    old_artery = all_pred_data[0, 6]
                    cap = all_gt_data[0, 5]
                    fen = all_gt_data[0, 6] # 医生版

                elif self.mode == 'our_gt':
                    old_artery = all_pred_data[0, 6]
                    label_cap = all_pred_data[0, 0]
                    label_fen = all_pred_data[0, 2]
               
                elif self.mode == 'our_semantic':
                    old_artery = all_pred_data[0, 6]
                    label_cap_semantic = all_pred_data[0, 1]
                    label_fen_semantic = all_pred_data[0, 3]

                elif self.mode == 'our_pred':
                    new_artery = all_pred_data[0, 5]
                    pred_cap = all_pred_data[0, 4]
                    label_fen = all_pred_data[0, 2]
                
                # 训练期间 数据增强
                if self.phase == 'train' and np.random.uniform() < 0.5:
                    axis = np.random.randint(0, 3)

                    artery_data = np.flip(artery_data, axis=axis)
                    ace = np.flip(ace, axis=axis)
                    venous_data = np.flip(venous_data, axis=axis)

                    # venous_data, artery_data = augment_gaussian_noise(venous_data, artery_data, p_per_channel=1) # 两种不同的
                    venous_data, artery_data = augment_gaussian_blur(venous_data, artery_data, p_per_channel=1)
                    # venous_data, artery_data = augment_brightness_multiplicative(venous_data, artery_data, p_per_channel=1)
                    # venous_data = augment_gamma(venous_data, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, retain_stats=True)
                    # artery_data = augment_gamma(artery_data, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, retain_stats=True)
                    # venous_data = augment_contrast(venous_data, contrast_range=(0.75, 1.25), preserve_range=True)
                    # artery_data = augment_contrast(artery_data, contrast_range=(0.75, 1.25), preserve_range=True)

                    if self.mode == 'ori':
                        cap = np.flip(cap, axis=axis)
                        fen = np.flip(fen, axis=axis)
                        old_artery = np.flip(old_artery, axis=axis)

                    elif self.mode == 'our_gt':
                        old_artery = np.flip(old_artery, axis=axis)
                        label_cap = np.flip(label_cap, axis=axis)
                        label_fen = np.flip(label_fen, axis=axis)

                    elif self.mode == 'our_semantic':
                        old_artery = np.flip(old_artery, axis=axis)
                        label_cap_semantic = np.flip(label_cap_semantic, axis=axis)
                        label_fen_semantic = np.flip(label_fen_semantic, axis=axis)

                    elif self.mode == 'our_pred':
                        label_fen = np.flip(label_fen, axis=axis)
                        pred_cap = np.flip(pred_cap, axis=axis)
                        new_artery = np.flip(new_artery, axis=axis)
                    

                    if self.do_rotation and np.random.uniform() < 0.5:
                        k = np.random.randint(0, 3)
                        rot_axis_th = np.random.randint(0, 3)
                        rot_axis_list = [(0,1),(1,2),(0,2)]
                        rot_axis = rot_axis_list[rot_axis_th]

                        artery_data = np.rot90(artery_data, k, rot_axis)
                        ace = np.rot90(ace, k, rot_axis)
                        venous_data = np.rot90(venous_data, k, rot_axis)

                        if self.mode == 'ori':
                            old_artery = np.rot90(old_artery, k, rot_axis)
                            cap = np.rot90(cap, k, rot_axis)
                            fen = np.rot90(fen, k, rot_axis)

                        elif self.mode == 'our_gt':
                            old_artery = np.rot90(old_artery, k, rot_axis)
                            label_cap = np.rot90(label_cap, k, rot_axis) # 经过腐蚀膨胀处理的
                            label_fen = np.rot90(label_fen, k, rot_axis)

                        elif self.mode == 'our_semantic':
                            old_artery = np.rot90(old_artery, k, rot_axis)
                            label_cap_semantic = np.rot90(label_cap_semantic, k, rot_axis)
                            label_fen_semantic = np.rot90(label_fen_semantic, k, rot_axis)

                        elif self.mode == 'our_pred':
                            pred_cap = np.rot90(pred_cap, k, rot_axis)
                            new_artery = np.rot90(new_artery, k, rot_axis)
                            label_fen = np.rot90(label_fen, k, rot_axis)
                        
                # 数据增强后
                if self.mode == 'ori':
                    all_data[j, :1] = artery_data[None][None]
                    all_data[j, 1:2] = old_artery[None][None]
                    all_data[j, 2:3] = ace[None][None]

                    all_data[j, 3:4] = venous_data[None][None]
                    all_data[j, 4:5] = cap[None][None]
                    all_data[j, 5:6] = fen[None][None]
                elif self.mode == 'our_gt':
                    all_data[j, :1] = artery_data[None][None]
                    all_data[j, 1:2] = old_artery[None][None]
                    all_data[j, 2:3] = ace[None][None]

                    all_data[j, 3:4] = venous_data[None][None]
                    all_data[j, 4:5] = label_cap[None][None]
                    all_data[j, 5:6] = label_fen[None][None] 
                elif self.mode == 'our_semantic':
                    all_data[j, :1] = artery_data[None][None]
                    all_data[j, 1:2] = old_artery[None][None]
                    all_data[j, 2:3] = ace[None][None]

                    all_data[j, 3:4] = venous_data[None][None]
                    all_data[j, 4:5] = label_cap_semantic[None][None]
                    all_data[j, 5:6] = label_fen_semantic[None][None] 
                elif self.mode == 'our_pred':
                    all_data[j, :1] = artery_data[None][None]
                    all_data[j, 1:2] = new_artery[None][None]
                    all_data[j, 2:3] = ace[None][None]

                    all_data[j, 3:4] = venous_data[None][None]
                    all_data[j, 4:5] = pred_cap[None][None]
                    all_data[j, 5:6] = label_fen[None][None] 
                else:
                    raise ValueError

            # 不使用属性
            else:
                artery_data = all_gt_data[0, 0]
                venous_data = all_gt_data[0, 4]

                if self.phase == 'train' and np.random.uniform() < 0.5:
                    axis = np.random.randint(0, 3)
                    artery_data = np.flip(artery_data, axis=axis)
                    venous_data = np.flip(venous_data, axis=axis)
                    if self.do_rotation and np.random.uniform() < 0.5:
                        k = np.random.randint(0, 3)
                        rot_axis_th = np.random.randint(0, 3)
                        rot_axis_list = [(0,1),(1,2),(0,2)]
                        rot_axis = rot_axis_list[rot_axis_th]

                        artery_data = np.rot90(artery_data, k, rot_axis)
                        venous_data = np.rot90(venous_data, k, rot_axis)

                all_data[j, :1] = artery_data[None][None]
                all_data[j, 1:2] = venous_data[None][None]


            # ['size', 'ace', 'cap+', 'cap++', 'fen+', 'fen++', 'radio', 'num']
            # size_norm = (np.max(crop_size_plus) - self.crop_size_mean) / self.crop_size_std
            # ace_norm = (int(have_ace_or_not) - self.ace_mean) / self.ace_std
            # cap1_norm = (float(pred_cap1) - self.cap1_mean) / self.cap1_std
            # cap2_norm = (float(pred_cap2) - self.cap2_mean) / self.cap2_std
            # fen1_norm = (float(our_gt_fen1) - self.fen1_mean) / self.fen1_std
            # fen2_norm = (float(our_gt_fen2) - self.fen2_mean) / self.fen2_std
            # radio_norm = (float(pred_radio) - self.radio_mean) / self.radio_std
            # num_norm = (float(pred_num) - self.num_mean) / self.num_std

            if self.mode == 'our_pred':
                size_norm = (np.max(crop_size_plus) - self.crop_size_min) / (self.crop_size_max - self.crop_size_min)
                ace_norm = (int(have_ace_or_not) - self.ace_min) / (self.ace_max - self.ace_min)
                cap1_norm = (float(pred_cap1) - self.cap1_min) / (self.cap1_max - self.cap1_min)
                cap2_norm = (float(pred_cap2) - self.cap2_min) / (self.cap2_max - self.cap2_min)
                fen1_norm = (float(our_gt_fen1) - self.fen1_min) / (self.fen1_max - self.fen1_min)
                fen2_norm = (float(our_gt_fen2) - self.fen2_min) / (self.fen2_max - self.fen2_min)
                radio_norm = (float(pred_radio) - self.radio_min) / (self.radio_max - self.radio_min)
                num_norm = (float(pred_num) - self.num_min) / (self.num_max - self.num_min)
                # attrs_feature_each = [np.max(crop_size_plus), int(have_ace_or_not), float(pred_cap1), float(pred_cap2),
                #                     float(our_gt_fen1), float(our_gt_fen2), float(pred_radio), float(pred_num)]
                attrs_feature_each = [size_norm, ace_norm, cap1_norm, cap2_norm,
                                    fen1_norm, fen2_norm, radio_norm, num_norm]
            elif self.mode == 'ori':
                size_norm = (np.max(crop_size_plus) - self.crop_size_min) / (self.crop_size_max - self.crop_size_min)
                ace_norm = (int(have_ace_or_not) - self.ace_min) / (self.ace_max - self.ace_min)
                cap1_norm = (float(ori_cap1) - self.cap1_min) / (self.cap1_max - self.cap1_min)
                cap2_norm = (float(ori_cap2) - self.cap2_min) / (self.cap2_max - self.cap2_min)
                fen1_norm = (float(ori_fen1) - self.fen1_min) / (self.fen1_max - self.fen1_min)
                fen2_norm = (float(ori_fen2) - self.fen2_min) / (self.fen2_max - self.fen2_min)
                radio_norm = (float(ori_radio) - self.radio_min) / (self.radio_max - self.radio_min)
                num_norm = (float(ori_num) - self.num_min) / (self.num_max - self.num_min)
                # attrs_feature_each = [np.max(crop_size_plus), int(have_ace_or_not), float(pred_cap1), float(pred_cap2),
                #                     float(our_gt_fen1), float(our_gt_fen2), float(pred_radio), float(pred_num)]
                attrs_feature_each = [size_norm, ace_norm, cap1_norm, cap2_norm,
                                    fen1_norm, fen2_norm, radio_norm, num_norm]
            elif self.mode == 'our_gt':
                size_norm = (np.max(crop_size_plus) - self.crop_size_min) / (self.crop_size_max - self.crop_size_min)
                ace_norm = (int(have_ace_or_not) - self.ace_min) / (self.ace_max - self.ace_min)
                cap1_norm = (float(our_gt_cap1) - self.cap1_min) / (self.cap1_max - self.cap1_min)
                cap2_norm = (float(our_gt_cap2) - self.cap2_min) / (self.cap2_max - self.cap2_min)
                fen1_norm = (float(our_gt_fen1) - self.fen1_min) / (self.fen1_max - self.fen1_min)
                fen2_norm = (float(our_gt_fen2) - self.fen2_min) / (self.fen2_max - self.fen2_min)
                radio_norm = (float(ori_radio) - self.radio_min) / (self.radio_max - self.radio_min)
                num_norm = (float(ori_num) - self.num_min) / (self.num_max - self.num_min)
                # attrs_feature_each = [np.max(crop_size_plus), int(have_ace_or_not), float(pred_cap1), float(pred_cap2),
                #                     float(our_gt_fen1), float(our_gt_fen2), float(pred_radio), float(pred_num)]
                attrs_feature_each = [size_norm, ace_norm, cap1_norm, cap2_norm,
                                    fen1_norm, fen2_norm, radio_norm, num_norm]

                        
            # j level
            label_list.append(mvi)
            resize_factor_list.append(resize_factor)
            only_id = line[0].split('/')[-1][:-4]
            name_list.append(str(only_id))

            center_point_list.append(center_point)
            crop_size_plus_list.append(crop_size_plus)

            attrs_feature_list.append(attrs_feature_each)

        label_list = np.array(label_list)
        resize_factor_list = np.array(resize_factor_list)
        name_list = np.array(name_list)
        center_point_list = np.array(center_point_list)
        crop_size_plus_list = np.array(crop_size_plus_list)

        # magin_array
        if self.args.triplet_flag:
            for i in range(len(lines_plus)): # anchor
                for j in range(len(lines_plus)): #p
                    for k in range(len(lines_plus)): #n
                        # hard 0.1
                        difference_ij = self.attrs_to_difference_fn(attrs_feature_list[i], attrs_feature_list[j], self.correlation_coeff, n_rank=5, alpha=0.1) #ap
                        difference_ik = self.attrs_to_difference_fn(attrs_feature_list[i], attrs_feature_list[k], self.correlation_coeff, n_rank=5, alpha=0.1) #an
                        magin_array[i,j,k] = difference_ik - difference_ij #ap属性相似，an属性不相似，magin变大 # magin = difference_ik - difference_ij
                        # magin_array[i,j,k] = 1.0# 0.1,0.3,1.0,0.5
                        
            # magin_array[magin_array<0.3] = 0.3
            # magin_array[magin_array<0.1] = 0.1
            # magin_array[magin_array>2.0] = 2.0
            # magin_array[magin_array>0]
            # magin_array[magin_array<0] = magin_array[magin_array<0]/2.0
            magin_array = torch.abs(magin_array)
            
        
        if self.use_bio_marker:
            cap1_list, cap2_list, cap3_list = np.array(cap1_list), np.array(cap2_list), np.array(cap3_list)
            fen1_list, fen2_list, fen3_list = np.array(fen1_list), np.array(fen2_list), np.array(fen3_list)
            num_list = np.array(num_list)
            radio_list = np.array(radio_list)
            have_ace_or_not_list = np.array(have_ace_or_not_list)

            data_dict = {'all_data': all_data, 'label': label_list, 'resize_factor': resize_factor_list, 'sample_finish_flag': sample_finish_flag, 'name_list': name_list,
            'cap1_list': cap1_list, 'cap2_list': cap2_list, 'cap3_list': cap3_list, 'fen1_list': fen1_list, 'fen2_list': fen2_list, 'fen3_list': fen3_list, 'magin_array': magin_array,
            'num_list': num_list, 'radio_list': radio_list, 'have_ace_or_not_list': have_ace_or_not_list, 'center_point_list': center_point_list, 'crop_size_plus_list': crop_size_plus_list}
        else:
            data_dict = {'all_data': all_data, 'label': label_list, 'resize_factor': resize_factor_list, 'sample_finish_flag': sample_finish_flag, 'name_list': name_list,
            'center_point_list': center_point_list, 'crop_size_plus_list': crop_size_plus_list, 'magin_array': magin_array}


        for key in ['all_data', 'label']:
            if isinstance(data_dict[key], np.ndarray):
                data_dict[key] = torch.from_numpy(data_dict[key]).float().contiguous()
            elif isinstance(data_dict[key], (list, tuple)) and all([isinstance(i, np.ndarray) for i in data_dict[key]]):
                data_dict[key] = [torch.from_numpy(i).float().contiguous() for i in data_dict[key]]

        return data_dict

        
    def __len__(self):
        return len(self._lines)//self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_train_batch_V3()

    def next(self):
        return self.__next__()



if __name__ == '__main__':
    pass