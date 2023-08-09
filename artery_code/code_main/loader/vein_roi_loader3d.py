import numpy as np
from collections import OrderedDict
import torch
import copy
import random
import os
from augmentations.augment_spatial import augment_spatial, get_patch_size
from augmentations.augment_more import *
from utils.file_ops import *
import cv2


class DataLoader3D(object):

    def __init__(self, Basic_Args, split_flag):
        self.args = Basic_Args
        self.split_flag = split_flag
        self.patch_size = self.args.patch_size
        self.batch_size = self.args.batch_size
        self.oversample_foreground_percent = self.args.oversample_foreground_percent
        if self.split_flag == 'train':
            self.training = True
        else:
            self.training = False

        self.lists = self.prepare_data(self.args)

        # 旋转数据增强
        rotation_x = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        rotation_y = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        rotation_z = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

        self.rotation_x = rotation_x
        self.rotation_y = rotation_y
        self.rotation_z = rotation_z

        if self.training:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, rotation_x, rotation_y, rotation_z, (0.85, 1.25))
        else:
            self.basic_generator_patch_size = self.patch_size

        self.need_to_pad = (np.array(self.basic_generator_patch_size) - np.array(self.patch_size)).astype(int)
        self.data_shape, self.seg_shape = self.determine_shapes()


    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))


    def determine_shapes(self):
        data_shape = (self.batch_size, 1, *self.basic_generator_patch_size)
        seg_shape = (self.batch_size, 1, *self.basic_generator_patch_size)
        return data_shape, seg_shape


    def generate_train_batch(self):
        self.pad_kwargs_data = OrderedDict()
    
        selected_lists = []
        line_selected_th_list = np.random.choice(len(self.lists), self.batch_size, False, None) # False:每个都是不同的patient
        for line_selected_th in line_selected_th_list:
            selected_lists.append(self.lists[line_selected_th])

        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)

        for j, each_list in enumerate(selected_lists):
            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            # print(each_list)
            ori_img = np.load(each_list[0], 'r')[None]
            ori_artery = np.load(each_list[1], 'r')[None]
            assert ori_img.shape == ori_artery.shape
            assert len(ori_img.shape) == len(ori_artery.shape) == 4, print(ori_img.shape, ori_artery.shape)

            case_all_data = ori_img
            case_all_data = np.concatenate((case_all_data, ori_artery), axis=0)


            need_to_pad = self.need_to_pad
            for d in range(3):
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.basic_generator_patch_size[d]:
                    need_to_pad[d] = self.basic_generator_patch_size[d] - case_all_data.shape[d + 1]

            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2 
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.basic_generator_patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.basic_generator_patch_size[1]
            lb_z = - need_to_pad[2] // 2
            ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.basic_generator_patch_size[2]
            
            if not force_fg:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
            else:
                class_locations = each_list[-1]
                foreground_classes = np.array([i for i in class_locations.keys() if len(class_locations[i]) != 0])
                foreground_classes = foreground_classes[foreground_classes > 0]

                if len(foreground_classes) == 0:
                    selected_class = None
                    voxels_of_that_class = None
                    print('case does not contain any foreground classes', i)
                else:
                    to_choose_class = [1]
                    while len(to_choose_class) > 0:
                        selected_class = np.random.choice(to_choose_class)
                        voxels_of_that_class = class_locations[selected_class]
                        if len(voxels_of_that_class) != 0:
                            break
                        else:
                            to_choose_class.remove(selected_class)
                if len(voxels_of_that_class) != 0 :
                    selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                    bbox_x_lb = max(lb_x, selected_voxel[0] - self.basic_generator_patch_size[0] // 2)
                    bbox_y_lb = max(lb_y, selected_voxel[1] - self.basic_generator_patch_size[1] // 2)
                    bbox_z_lb = max(lb_z, selected_voxel[2] - self.basic_generator_patch_size[2] // 2)
                else:
                    bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                    bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                    bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

            bbox_x_ub = bbox_x_lb + self.basic_generator_patch_size[0]
            bbox_y_ub = bbox_y_lb + self.basic_generator_patch_size[1]
            bbox_z_ub = bbox_z_lb + self.basic_generator_patch_size[2]

            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)
            valid_bbox_z_lb = max(0, bbox_z_lb)
            valid_bbox_z_ub = min(shape[2], bbox_z_ub)

            case_all_data = np.copy(case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub, valid_bbox_y_lb:valid_bbox_y_ub, valid_bbox_z_lb:valid_bbox_z_ub])

            data[j, 0] = np.pad(case_all_data[:1], ((0, 0), 
                                                  (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                  (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                  (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                             "constant", **self.pad_kwargs_data)

            seg[j, 0] = np.pad(case_all_data[1:2], ((0, 0),
                                                    (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                    (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                    (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                               'constant', **{'constant_values': -1})


        
        # 使用空间数据增强
        if self.training:
            data, seg = augment_spatial(data, seg, self.patch_size, do_rotation=True, angle_x=self.rotation_x, angle_y=self.rotation_y, \
                                        angle_z=self.rotation_z, do_scale=True, scale=(0.7, 1.4), do_dummy_2D_aug=False)

        data_dict = {'data': data, 'seg': seg}
        key_list = ['data', 'seg']

        # 使用更多的数据增强
        if self.training:
            ignore_axes = None
            tr_transforms = []
            tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
            tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                        p_per_channel=0.5))
            tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15)) # channel-wise | for not ted
            tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15)) # channel-wise | for not ted
            tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                                p_per_channel=0.5,
                                                                order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                                ignore_axes=ignore_axes))
            tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1)) # invert gamma =True
            tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3)) # invert gamma =False
            tr_transforms.append(MirrorTransform((0, 1, 2)))
            tr_transforms = Compose(tr_transforms)
            data_dict = tr_transforms(data_dict)
        
        seg = data_dict['seg']
        seg[seg == -1] = 0
        data_dict['seg'] = seg

        for key in key_list:
            if isinstance(data_dict[key], np.ndarray):
                data_dict[key] = torch.from_numpy(data_dict[key]).float().contiguous()
            elif isinstance(data_dict[key], (list, tuple)) and all([isinstance(i, np.ndarray) for i in data_dict[key]]):
                data_dict[key] = [torch.from_numpy(i).float().contiguous() for i in data_dict[key]]

        return data_dict


    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)
        if add_timestamp:
            args = ("%s:" % dt_object, *args)
        if also_print_to_console:
            print(*args)


    def __iter__(self):
        return self


    def __next__(self):
        return self.generate_train_batch()


    def next(self):
        return self.__next__()
        

    def prepare_data(self, args):
        '''注意血管的分割要使用动脉期'''
        root_path = args.data_path
        ori_samples = os.listdir(root_path)
        ori_samples = [join(root_path, each_sample, each_sample+'_roi.npy') for each_sample in ori_samples if os.path.exists(join(root_path, each_sample, each_sample+'_roi.npy'))]
        ori_samples.sort()

        samples = []
        lists = []
        samples_id = []
        final_lists = []
        for each_sample in ori_samples:
            if not os.path.exists(each_sample):
                  continue
            each_sample_id = each_sample.split('/')[-2]
            samples_id.append(each_sample_id)
            samples.append(each_sample)
            # data = sitk.GetArrayFromImage(sitk.ReadImage(each_sample.replace('hcc_surg_artery_roi.nii', 'hcc_surg_artery.nii')))[None]
            # tmv_label = sitk.GetArrayFromImage(sitk.ReadImage(each_sample))[None]
            # maybe_mkdir_p(root_path.replace('artery_raw_data','artery_npy_data/'+str(each_sample_id)))
            # if not os.path.exists(root_path.replace('artery_raw_data','artery_npy_data')+'/'+str(each_sample_id)+'/'+each_sample_id+'_tmv.npy'):
            #     np.save(root_path.replace('artery_raw_data','artery_npy_data')+'/'+str(each_sample_id)+'/'+each_sample_id+'_tmv.npy', tmv_label)
            #     np.save(root_path.replace('artery_raw_data','artery_npy_data')+'/'+str(each_sample_id)+'/'+each_sample_id+'_data.npy', data)
            # path_list = [root_path.replace('artery_raw_data','artery_npy_data')+'/'+str(each_sample_id)+'/'+each_sample_id+'_data.npy', root_path.replace('artery_raw_data','artery_npy_data')+'/'+str(each_sample_id)+'/'+each_sample_id+'_tmv.npy']
            path_list = [each_sample.replace('roi','img_roi'), each_sample]
            lists.append(path_list)


        train_num = int(len(samples) * args.split_prop)
        val_num = len(samples) - train_num

        # only of record
        all_trainable_list =  open(join('./tednet/artery_code/code_main/lists', args.task_name + '_train_split.txt'), 'w')
        all_valable_list =  open(join('./tednet/artery_code/code_main/lists', args.task_name + '_val_split.txt'), 'w')
        count = 0
        for each_sample in samples:
            each_sample_id = each_sample.split('/')[-2]
            if count < train_num:
                all_trainable_list.write(each_sample_id + '\n')
                count += 1
            else:
                all_valable_list.write(each_sample_id + '\n')
        

        intensityproperties_root_path = './tednet/artery_code/code_main/lists'
        maybe_mkdir_p(intensityproperties_root_path)

        if not os.path.exists(join(intensityproperties_root_path, args.task_name + '_artery_intensity_' + self.split_flag + '.pkl')):
            first_build = True
            properties = OrderedDict()
        else:
            first_build = False
            with open(join(intensityproperties_root_path, args.task_name + '_artery_intensity_' + self.split_flag + '.pkl'), 'rb') as f:
                properties = pickle.load(f)

        for th in range(len(lists)):
            if self.split_flag == 'train' and th >= count:
                continue
            if self.split_flag != 'train' and th < count:
                continue
            patient = lists[th]
            artery_label = np.load(patient[1], 'r')
            only_id = samples_id[th]
            assert only_id in patient[1]
            if first_build:
                rndst = np.random.RandomState(1234)
                class_locs = {}
                all_locs = np.argwhere(artery_label == 1) #[[]]
                if len(all_locs) == 0:
                    ValueError
                target_num_samples = min(10000, len(all_locs))
                target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * 0.01)))
                selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
                class_locs[1] = selected
                properties[only_id] = class_locs
            else:
                class_locs = properties[only_id]

            each_patient_data_list = patient + [class_locs]
            final_lists.append(each_patient_data_list)
        
        if first_build:
            with open(join(intensityproperties_root_path, args.task_name + '_artery_intensity_' + self.split_flag + '.pkl'), 'wb') as f:
                pickle.dump(properties, f)

        return final_lists
