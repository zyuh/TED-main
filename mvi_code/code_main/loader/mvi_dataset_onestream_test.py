import csv
import numpy as np
import torch
import random
from collections import OrderedDict
import os
# from augmentations.augment_spatial import augment_spatial, augment_spatial_mvi_1, augment_spatial_mvi_2
from augmentations.augment_more import *
from utils.file_ops import *
from skimage.transform import resize
from multiprocessing import Pool
import threading
from threading import Lock,Thread
from scipy.ndimage import gaussian_filter
import torch.backends.cudnn as cudnn




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



class MVI_Dataloader_mvi_test(object):
    def __init__(self, Basic_Args):
        self.args = Basic_Args

        self.mode = 'gt'
        self.patch_size = [64, 64, 64]
        self.batch_size = 4

        self.lines, self.line_head = self.prepare_dataV4()
        self._lines = copy.deepcopy(self.lines)

        self.use_attribute = self.args.use_attr
        self.use_bio_marker = self.args.use_bio_marker

        if self.args.use_ourtriplet:
            # self.metrics_attrs = ['cap+', 'cap++', 'fen+', 'fen++', 'num', 'radio', 'ace', 'size'] # 
            self.correlation_coeff = [0.40,   0.33,   0.12,   0.13,  0.30,  0.25,   0.21,   0.5] # CODE/myxgboost.py
            self.correlation_coeff = torch.tensor(self.correlation_coeff)
            self.attrs_to_difference_fn = get_difference_soft

        cap1_list, cap2_list, cap3_list, fen1_list, fen2_list, fen3_list,\
            radio_list, num_list, crop_size_plus_list = self.get_mean_std_for_each_attr()
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

        self.radio_max, self.radio_min = np.max(radio_list), np.min(radio_list)
        self.num_max, self.num_min = np.max(num_list), np.min(num_list)
        self.crop_size_max, self.crop_size_min = np.max(crop_size_plus_list), np.min(crop_size_plus_list)

        self.data_shape, self.data_shape_tmp, self.each_data_shape = self.determine_shapes()


    def get_mean_std_for_each_attr(self):
        root_path = self.args.data_path
        bio_marker_csv = root_path.replace(root_path.split('/')[-1], '/mvi_code/code_main/final_biomarkers.csv')
        cap1_list, cap2_list, cap3_list = [],[],[]
        fen1_list, fen2_list, fen3_list = [],[],[]
        radio_list, num_list = [],[]
        crop_size_plus_list = []

        with open(bio_marker_csv, 'r') as f:
            reader = csv.reader(f)
            if self.mode == 'gt':
                for row in reader:
                    if row[0]!='UID':
                        cap1_list.append(float(row[9]))
                        cap2_list.append(float(row[10]))
                        cap3_list.append(float(row[11]))
                        fen1_list.append(float(row[12]))
                        fen2_list.append(float(row[13]))
                        fen3_list.append(float(row[14]))
                        if row[23]!='None':
                            num_list.append(float(row[23])) # # new_artery
                        else:
                            num_list.append(float(0))
                        if row[24]!='None':
                            radio_list.append(float(row[24])) # new_artery
                        else:
                            radio_list.append(float(0))
                        crop_size_plus_list.append(float(row[29]))

            elif self.mode == 'pred':
                for row in reader:
                    if row[0]!='UID':
                        cap1_list.append(float(row[15]))
                        cap2_list.append(float(row[16]))
                        cap3_list.append(float(row[17]))
                        fen1_list.append(float(row[18]))
                        fen2_list.append(float(row[19]))
                        fen3_list.append(float(row[20]))
                        if row[25]!='None':
                            num_list.append(float(row[25])) # # new_artery
                        else:
                            num_list.append(float(0))
                        if row[26]!='None':
                            radio_list.append(float(row[26])) # new_artery
                        else:
                            radio_list.append(float(0))
                        crop_size_plus_list.append(float(row[29]))

        return cap1_list, cap2_list, cap3_list, fen1_list, fen2_list, fen3_list,\
                radio_list, num_list, crop_size_plus_list


    def prepare_dataV4(self):

        root_path = self.args.data_path
        all_sample_list = os.listdir(root_path)
        all_sample_list = [os.path.join(root_path, each_sample) for each_sample in all_sample_list if 'csv' not in each_sample]
        all_sample_list.sort()

        examples_list = [join(each_sample, each_sample.split('/')[-1]+'_prepare_mvi.npy') for each_sample in all_sample_list]
        lists = []

        for th in range(len(examples_list)):
            each_example_path = examples_list[th]
            each_pkl_path = each_example_path.replace('.npy', '.pkl')
            each_pkl = load_pickle(each_pkl_path)

            resize_factor = each_pkl['resize_factor']

            only_id = each_pkl_path.split('/')[-2]

            # each_patient_data_list = [each_example_path, each_pkl_path, resize_factor, mvi]
            each_patient_data_list = [each_example_path, each_pkl_path, resize_factor]
            bio_marker_csv = os.path.join(root_path, 'attr_biomarkers.csv')
            bio_marker = []
            head_plus = ['each_example_path', 'each_pkl_path', 'resize_factor', 'mvi', 'cap+', 'cap++', 'fen+', 'fen++', 'artery_num', 'artery_ratio', 'have_ace', 'max_size']
            with open(bio_marker_csv, 'r') as f:
                reader = csv.reader(f)
                for each_line in reader:
                    if each_line[0] == 'case_id': # 跳过表头
                        continue
                    if only_id == each_line[0]:
                        bio_marker = [None, each_line[1], each_line[2], each_line[4], each_line[5], each_line[7], each_line[8], each_line[9], each_line[10]]
                        break
            f.close()
            each_patient_data_list = each_patient_data_list + bio_marker 

            assert len(head_plus) == len(each_patient_data_list),  print(len(head_plus), len(each_patient_data_list))
            lists.append(each_patient_data_list)

        return lists, head_plus


    def determine_shapes(self):
        if self.use_attribute:
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

            all_data_read = np.load(line[0], 'r')

            resize_factor = float(line[2])
            # mvi = int(line[3])
            have_ace = int(line[-2])
            crop_size_plus = int(line[-1]) # max size
            have_ace_or_not_list.append(int(have_ace))

            cap1 = float(line[4])
            cap2 = float(line[5])
            fen1 = float(line[6])
            fen2 = float(line[7])
            artery_num = float(line[8])
            artery_radio = float(line[9])

            cap1_list.append((float(cap1) - self.cap1_min) / (self.cap1_max - self.cap1_min))
            cap2_list.append((float(cap2) - self.cap2_min) / (self.cap2_max - self.cap2_min))
            fen1_list.append((float(fen1) - self.fen1_min) / (self.fen1_max - self.fen1_min))
            fen2_list.append((float(fen2) - self.fen2_min) / (self.fen2_max - self.fen2_min))
            num_list.append((float(artery_num) - self.num_min) / (self.num_max - self.num_min))
            radio_list.append((float(artery_radio) - self.radio_min) / (self.radio_max - self.radio_min))

            all_data_ori = copy.deepcopy(all_data_read[:, :, 4:68, 4:68, 4:68])

            venous_data = all_data_ori[0, 0]
            artery_data = all_data_ori[0, 1]
                
            if self.use_attribute:
                artery_mask = all_data_ori[0, 6]
                cap_mask = all_data_ori[0, 3]
                fen_mask = all_data_ori[0, 4]
                ace_mask = all_data_ori[0, 7]
 
                all_data[j, :1] = artery_data[None][None]
                all_data[j, 1:2] = artery_mask[None][None]
                all_data[j, 2:3] = ace_mask[None][None]

                all_data[j, 3:4] = venous_data[None][None]
                all_data[j, 4:5] = cap_mask[None][None]
                all_data[j, 5:6] = fen_mask[None][None] 

            else:
                venous_data = all_data_ori[0, 0]
                artery_data = all_data_ori[0, 1]

                all_data[j, :1] = artery_data[None][None]
                all_data[j, 1:2] = venous_data[None][None]


            size_norm = (crop_size_plus - self.crop_size_min) / (self.crop_size_max - self.crop_size_min)
            ace_norm = int(have_ace)
            cap1_norm = (float(cap1) - self.cap1_min) / (self.cap1_max - self.cap1_min)
            cap2_norm = (float(cap2) - self.cap2_min) / (self.cap2_max - self.cap2_min)
            fen1_norm = (float(fen1) - self.fen1_min) / (self.fen1_max - self.fen1_min)

            fen2_norm = (float(fen2) - self.fen2_min) / (self.fen2_max - self.fen2_min)
            radio_norm = (float(artery_radio) - self.radio_min) / (self.radio_max - self.radio_min)
            num_norm = (float(artery_num) - self.num_min) / (self.num_max - self.num_min)
 
            attrs_feature_each = [size_norm, ace_norm, cap1_norm, cap2_norm,
                                fen1_norm, fen2_norm, radio_norm, num_norm]
 
            # j level
            # label_list.append(mvi)
            resize_factor_list.append(resize_factor)
            only_id = line[0].split('/')[-2]
            name_list.append(str(only_id))

            crop_size_plus_list.append(crop_size_plus)
            attrs_feature_list.append(attrs_feature_each)

        # label_list = np.array(label_list)
        resize_factor_list = np.array(resize_factor_list)
        name_list = np.array(name_list)
        crop_size_plus_list = np.array(crop_size_plus_list)

        # magin_array
        if self.args.use_ourtriplet:
            for i in range(len(lines_plus)): # anchor
                for j in range(len(lines_plus)): #p
                    for k in range(len(lines_plus)): #n
                        difference_ij = self.attrs_to_difference_fn(attrs_feature_list[i], attrs_feature_list[j], self.correlation_coeff, n_rank=5, alpha=0.1) #ap
                        difference_ik = self.attrs_to_difference_fn(attrs_feature_list[i], attrs_feature_list[k], self.correlation_coeff, n_rank=5, alpha=0.1) #an
                        magin_array[i,j,k] = 0.125 * (difference_ik - difference_ij) #ap属性相似，an属性不相似，magin变大 # magin = difference_ik - difference_ij

            magin_array = torch.abs(magin_array)
        
        if self.use_bio_marker:
            cap1_list, cap2_list = np.array(cap1_list), np.array(cap2_list)
            fen1_list, fen2_list = np.array(fen1_list), np.array(fen2_list)
            num_list = np.array(num_list)
            radio_list = np.array(radio_list)
            have_ace_or_not_list = np.array(have_ace_or_not_list)

            data_dict = {'all_data': all_data, 'label': label_list, 'resize_factor': resize_factor_list, 'sample_finish_flag': sample_finish_flag, 'name_list': name_list,
            'cap1_list': cap1_list, 'cap2_list': cap2_list, 'fen1_list': fen1_list, 'fen2_list': fen2_list, 'magin_array': magin_array,
            'num_list': num_list, 'radio_list': radio_list, 'have_ace_or_not_list': have_ace_or_not_list, 'max_size_list': crop_size_plus_list}
        else:
            data_dict = {'all_data': all_data, 'label': label_list, 'resize_factor': resize_factor_list, 'sample_finish_flag': sample_finish_flag, 'name_list': name_list,
            'magin_array': magin_array, 'max_size_list': crop_size_plus_list}


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