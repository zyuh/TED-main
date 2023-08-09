from utils.file_ops import *
from collections import OrderedDict
from multiprocessing import Pool
import numpy as np
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
import os
import copy



RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3
default_num_threads = 4


def resize_segmentation(segmentation, new_shape, order=3, cval=0):
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

def get_do_separate_z(spacing, anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z

def get_lowres_axis(new_spacing):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis


def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, cval=0, order_z=0):

    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False} #反混叠？
    dtype_data = data.dtype
    data = data.astype(float)
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape) # 不包含模态维度


    if np.any(shape != new_shape):
        if do_separate_z:
            # print("separate z, order in z is", order_z, "order inplane is", order)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]): #模态
                reshaped_data = []
                # 1: Bi-linear (default) #seg插值
                # 3: Bi-cubic  #img插值
                for slice_id in range(shape[axis]):  #slice
                    if axis == 0:
                        reshaped_data.append(resize_fn(data[c, slice_id], new_shape_2d, order, cval=cval, **kwargs)) #0/-1
                    elif axis == 1:
                        reshaped_data.append(resize_fn(data[c, :, slice_id], new_shape_2d, order, cval=cval, **kwargs))  #edge填充值，默认0
                    else:
                        reshaped_data.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order, cval=cval,
                                                       **kwargs))
                reshaped_data = np.stack(reshaped_data, axis)

                # 2d resample结束后该考虑第三个维度了
                if shape[axis] != new_shape[axis]:
                    # The following few lines are blatantly公然地 copied and modified from     sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        # mode: how the input array is extended beyond its boundaries
                        # 'nearest' (a a a a | a b c d | d d d d)
                        reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z, cval=cval,  #坐标映射 for img
                                                                   mode='nearest')[None])  #通过插值将输入数组映射到新坐标
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,#坐标映射 for seg
                                                cval=cval, mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl  #seg插值resample
                        reshaped_final_data.append(reshaped[None])

                else:
                    reshaped_final_data.append(reshaped_data[None])
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            # print("no separate z, order", order) #3img/1seg
            reshaped = []
            for c in range(data.shape[0]): #模态
                reshaped.append(resize_fn(data[c], new_shape, order, cval=cval, **kwargs)[None])
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data



def resample_patient(data, seg, original_spacing, target_spacing, order_data=3, order_seg=0, force_separate_z=False,
                     cval_data=0, cval_seg=-1, order_z_data=0, order_z_seg=0,
                     separate_z_anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD, aux_shape=None):  #3
    # for each case
    # TODO list
    assert not ((data is None) and (seg is None))
    if data is not None:
        assert len(data.shape) == 4, "data must be c x y z"
    if seg is not None:
        for each_seg in seg:
            if each_seg != 'None':
                assert len(each_seg.shape) == 4, "seg must be c x y z"

    if data is not None:
        shape = np.array(data[0].shape)
    else:
        shape = np.array(seg[0].shape)

    # resample 后的尺寸
    # print(original_spacing, target_spacing, shape)
    if aux_shape is not None:
        new_shape = np.array(list(aux_shape)).astype(int)
    else:
        new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)  #spacing to target_spacing(spacing统一)后的size

    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(original_spacing)
        else:
            axis = None
    else: #None
        if get_do_separate_z(original_spacing, separate_z_anisotropy_threshold):  #original_spacing是否有相差3倍以上的
            do_separate_z = True
            axis = get_lowres_axis(original_spacing)  #spacing最大的轴
        elif get_do_separate_z(target_spacing, separate_z_anisotropy_threshold): #target_spacing是否有相差3倍以上的
            do_separate_z = True
            axis = get_lowres_axis(target_spacing)
        else:
            do_separate_z = False
            axis = None


    if axis is not None:
        if len(axis) == 3:
            # every axis has the spacing
            axis = (0, )
        elif len(axis) == 2:
            print("WARNING: axis has len 2, axis: %s, spacing: %s, target_spacing: %s" % (str(axis), original_spacing, target_spacing))
            do_separate_z = False
        else:
            pass

    if data is not None:
        # print(original_spacing, target_spacing, new_shape, do_separate_z, axis)
        # assert do_separate_z == True and axis == [0]
        # assert do_separate_z == False and axis == None
        data_reshaped = resample_data_or_seg(data, new_shape, False, axis, order_data, do_separate_z, cval=cval_data,
                                             order_z=order_z_data)
    else:
        data_reshaped = None

    seg_reshaped_list = []
    if seg is not None:
        for each_seg_th in range(len(seg)):
            each_seg = seg[each_seg_th]
            # 注意fen 和 cap 也应该和 new_shape一致
            if each_seg != 'None':
                if each_seg_th==1 or each_seg_th==2: # cap or fen
                    do_separate_z_cap = True
                    axis_cap = [0]
                    seg_reshaped = resample_data_or_seg(each_seg, new_shape, True, axis_cap, order_seg, do_separate_z_cap, cval=cval_seg,
                                                        order_z=order_z_seg)
                else:
                    assert do_separate_z == False and axis == None
                    seg_reshaped = resample_data_or_seg(each_seg, new_shape, True, axis, order_seg, do_separate_z, cval=cval_seg,
                                                        order_z=order_z_seg)
                seg_reshaped_list.append(seg_reshaped)
            else:
                seg_reshaped_list.append('None')
    else:
        seg_reshaped = None
    return data_reshaped, seg_reshaped_list



class GenericPreprocessor(object):
    def __init__(self, normalization_scheme_per_modality, use_nonzero_mask, transpose_forward: (tuple, list), intensityproperties=None):
        self.transpose_forward = transpose_forward
        self.intensityproperties = intensityproperties
        self.normalization_scheme_per_modality = normalization_scheme_per_modality
        self.use_nonzero_mask = use_nonzero_mask

        self.resample_separate_z_anisotropy_threshold = RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD # 3

    @staticmethod
    def load_cropped(cropped_output_dir, case_identifier):
        data = np.load(case_identifier)['data'].astype(np.float32)
        seg_list = []
        if 'imagesTr' in case_identifier:
            roi_path = case_identifier.replace('imagesTr', 'labels_roi/train')
            fen_path = case_identifier.replace('imagesTr', 'labels_fen/train')
            cap_path = case_identifier.replace('imagesTr', 'labels_cap/train')
            ace_path = case_identifier.replace('imagesTr', 'labels_ACE/train')
            artery_path = case_identifier.replace('imagesTr', 'labels_artery_new/train')
        elif 'imagesTv' in case_identifier:
            roi_path = case_identifier.replace('imagesTv', 'labels_roi/valid')
            fen_path = case_identifier.replace('imagesTv', 'labels_fen/valid')
            cap_path = case_identifier.replace('imagesTv', 'labels_cap/valid')
            ace_path = case_identifier.replace('imagesTv', 'labels_ACE/valid')
            artery_path = case_identifier.replace('imagesTv', 'labels_artery_new/valid')
        elif 'imagesTs' in case_identifier:
            roi_path = case_identifier.replace('imagesTs', 'labels_roi/test')
            fen_path = case_identifier.replace('imagesTs', 'labels_fen/test')
            cap_path = case_identifier.replace('imagesTs', 'labels_cap/test')
            ace_path = case_identifier.replace('imagesTs', 'labels_ACE/test')
            artery_path = case_identifier.replace('imagesTs', 'labels_artery_new/test')
        else:
            ValueError
        path_list = [roi_path, fen_path, cap_path, ace_path, artery_path]
        for path_th in range(len(path_list)):
            path = path_list[path_th]
            if os.path.exists(path):
                each_laebl = np.load(path)['data']
                seg_list.append(each_laebl)
            else:
                seg_list.append('None')

        case_identifier_yes = case_identifier.split('/')[-1][:-4]
        pkl_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/cropped_data/cropped_pkl_all'
        with open(os.path.join(pkl_path, "%s.pkl" % case_identifier_yes), 'rb') as f:
            properties = pickle.load(f)
        return data, seg_list, properties, path_list

    def resample_and_normalize(self, data, target_spacing, properties, seg=None, force_separate_z=None):
        # for each case # 使用时，force_seperate_z为 None 

        # target_spacing is already transposed, properties["original_spacing"] is not so we need to transpose it!
        # data, seg are already transposed. Double check this using the properties
        original_spacing_transposed = np.array(properties["original_spacing"])[self.transpose_forward]
        before = {
            'spacing': properties["original_spacing"],
            'spacing_transposed': original_spacing_transposed,
            'data.shape (data is transposed)': data.shape
        }

        # remove nans
        data[np.isnan(data)] = 0# nan to 0

        ''' seg 为 list'''
        data, seg = resample_patient(data, seg, np.array(original_spacing_transposed), target_spacing, 3, 1,
                                     force_separate_z=force_separate_z, order_z_data=0, order_z_seg=0,
                                     separate_z_anisotropy_threshold=self.resample_separate_z_anisotropy_threshold)

        each_seg_shape = seg[0].shape
        for each_seg in seg:
            if each_seg!='None':
                assert np.prod(each_seg.shape) ==  np.prod(each_seg_shape)
                each_seg[each_seg < -1] = 0


        after = {
            'spacing': target_spacing,
            'data.shape (data is resampled)': data.shape
        }
        print("before:", before, "\nafter: ", after, "\n")

        # if seg is not None:  # hippocampus 243 has one voxel with -2 as label. wtf?
        #     seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        use_nonzero_mask = self.use_nonzero_mask

        assert len(self.normalization_scheme_per_modality) == len(data), "self.normalization_scheme_per_modality " \
                                                                         "must have as many entries as data has " \
                                                                         "modalities"
        assert len(self.use_nonzero_mask) == len(data), "self.use_nonzero_mask must have as many entries as data" \
                                                        " has modalities"
        '''对data进行标准化'''

        artery_data = copy.deepcopy(data)
        for c in range(len(artery_data)):
            scheme = self.normalization_scheme_per_modality[c]
            if scheme == "CT":
                # clip to lb(lower_bound) and ub(upper_bound) from train artery_data foreground and use foreground mn and sd from training artery_data
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                mean_intensity = self.intensityproperties[c]['mean_artery']
                std_intensity = self.intensityproperties[c]['sd_artery']
                lower_bound = self.intensityproperties[c]['percentile_00_5_artery']
                upper_bound = self.intensityproperties[c]['percentile_99_5_artery']
                artery_data[c] = np.clip(artery_data[c], lower_bound, upper_bound)
                # print('mean_intensity: {}, std_intensity: {} == do not clip 0.05 to 99.5 == '.format(mean_intensity, std_intensity))
                print('mean_intensity: {}, std_intensity: {}'.format(mean_intensity, std_intensity))
                artery_data[c] = (artery_data[c] - mean_intensity) / std_intensity
                if use_nonzero_mask[c]: #如果是CT，则为False
                    artery_data[c][seg[-1] < 0] = 0
        # ========
        roi_data = copy.deepcopy(data)
        for c in range(len(roi_data)):
            scheme = self.normalization_scheme_per_modality[c]
            if scheme == "CT":
                # clip to lb(lower_bound) and ub(upper_bound) from train roi_data foreground and use foreground mn and sd from training roi_data
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                mean_intensity = self.intensityproperties[c]['mean_roi']
                std_intensity = self.intensityproperties[c]['sd_roi']
                lower_bound = self.intensityproperties[c]['percentile_00_5_roi']
                upper_bound = self.intensityproperties[c]['percentile_99_5_roi']
                roi_data[c] = np.clip(roi_data[c], lower_bound, upper_bound)
                # print('mean_intensity: {}, std_intensity: {} == do not clip 0.05 to 99.5 == '.format(mean_intensity, std_intensity))
                print('mean_intensity: {}, std_intensity: {}'.format(mean_intensity, std_intensity))
                roi_data[c] = (roi_data[c] - mean_intensity) / std_intensity
                if use_nonzero_mask[c]: #如果是CT，则为False
                    roi_data[c][seg[-1] < 0] = 0

        data = [roi_data.astype(np.float32), artery_data.astype(np.float32)]

        return data, seg, properties

    def _run_internal(self, target_spacing, case_identifier, output_folder_stage, cropped_output_dir, force_separate_z,
                      all_classes_list):
        # 使用时，force_seperate_z为 None #TODO

        # case_identifier已经改成case了
        data, seg_list, properties, path_list = self.load_cropped(cropped_output_dir, case_identifier) #根据标识符进行加载npz和pkl,seglist

        # transfoward是在cropped之后计算的
        data = data.transpose((0, *[i + 1 for i in self.transpose_forward])) #[0,1,2]

        seg = []
        assert len(seg_list) == 5
        for each_seg in seg_list:
            if each_seg != 'None':
                each_seg = each_seg.transpose((0, *[i + 1 for i in self.transpose_forward])) #TODO list
                seg.append(each_seg)
            else:
                seg.append('None')
        

        #TODO list========
        data, seg, properties = self.resample_and_normalize(data, target_spacing,   
                                                            properties, seg, force_separate_z) #resample_and_normalize两个操作
        '''返回的data也是list 分别表示基于artery标准化 以及 基于roi标准化'''
        # all_data = np.vstack((data, seg)).astype(np.float32)  #TODO list


        '''采样结束后选取locations进行保存，list TODO'''
        # we need to find out where the classes are and sample some random locations
        # let's do 10.000 samples per class
        # seed this for reproducibility!
        num_samples = 10000
        min_percent_coverage = 0.01 # at least 1% of the class voxels need to be selected, otherwise it may be too sparse
        rndst = np.random.RandomState(1234)

        class_locs_list = []
        for each_seg_th in range(len(seg)):
            each_seg = seg[each_seg_th]
            all_classes = all_classes_list[each_seg_th]
            class_locs = {}
            for c in all_classes:
                all_locs = np.argwhere(each_seg == c) # 不包含背景
                if len(all_locs) == 0:
                    class_locs[c] = []
                    continue
                target_num_samples = min(num_samples, len(all_locs))
                target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

                selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
                class_locs[c] = selected
                print(c, target_num_samples)
            class_locs_list.append(class_locs)

        properties['class_locations'] = class_locs_list  #类别的大致分布位置？？？  #TODO list


        # case_identifier 是data的原始path
        case_id = case_identifier.split('/')[-1][:-4]
        aux_path = case_identifier.split('cropped_data')[-1].split('.npz')[0]
        maybe_mkdir_p(os.path.join(output_folder_stage + aux_path.split(case_id)[0] + '_roi'))
        print('saving... ', os.path.join(output_folder_stage + aux_path.split(case_id)[0] + '_roi', "%s.npy" % case_id))
        np.save(os.path.join(output_folder_stage + aux_path.split(case_id)[0] + '_roi', "%s.npy" % case_id), data[0].astype(np.float32))

        maybe_mkdir_p(os.path.join(output_folder_stage + aux_path.split(case_id)[0] + '_artery'))
        print('saving... ', os.path.join(output_folder_stage + aux_path.split(case_id)[0] + '_artery', "%s.npy" % case_id))
        np.save(os.path.join(output_folder_stage + aux_path.split(case_id)[0] + '_artery', "%s.npy" % case_id), data[1].astype(np.float32))
        # np.savez_compressed(os.path.join(self.output_folder + aux_path.split(case_id)[0], "%s.npz" % case_identifier), data=data)

        assert len(seg) == 5
        for th in range(len(seg)):
            each_seg = seg[th]
            each_seg_path = path_list[th]
            if each_seg!='None':
                each_seg_aux_path = each_seg_path.split('cropped_data')[-1].split('.npz')[0]
                maybe_mkdir_p(os.path.join(output_folder_stage + each_seg_aux_path.split(case_id)[0]))
                print('saving... ', os.path.join(output_folder_stage + each_seg_aux_path.split(case_id)[0], "%s.npy" % case_id))
                np.save(os.path.join(output_folder_stage + each_seg_aux_path.split(case_id)[0], "%s.npy" % case_id), each_seg)
                
        # print("saving: ", os.path.join(output_folder_stage, "%s.npy" % case_identifier))
        # np.save(os.path.join(output_folder_stage, "%s.npy" % case_identifier), all_data.astype(np.float32))

        pkl_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/cropped_data/cropped_pkl_all'
        with open(os.path.join(pkl_path, "%s.pkl" % case_id), 'wb') as f:  #npy文件????在训练器初始化unpack产生
            pickle.dump(properties, f)


    def run(self, target_spacings, input_folder_with_cropped_npz, output_folder, data_identifier,
            num_threads=default_num_threads, force_separate_z=None):
        # force_separate_z=None
        print("Initializing to run preprocessing")
        print("npz folder:", input_folder_with_cropped_npz) #cropped dir root path
        print("output_folder:", output_folder)


        train_set = join(input_folder_with_cropped_npz, 'imagesTr')
        valid_set = join(input_folder_with_cropped_npz, 'imagesTv')
        test_set = join(input_folder_with_cropped_npz, 'imagesTs')
        patient_identifiers_train = subfiles(train_set, join=True, suffix=".npz")
        patient_identifiers_valid = subfiles(valid_set, join=True, suffix=".npz")
        patient_identifiers_test = subfiles(test_set, join=True, suffix=".npz")
        list_of_cropped_npz_files = patient_identifiers_train + patient_identifiers_valid + patient_identifiers_test 
        assert len(list_of_cropped_npz_files)==477
        # list_of_cropped_npz_files = subfiles(input_folder_with_cropped_npz, True, None, ".npz", True) # TODO


        maybe_mkdir_p(output_folder) #preprocessed dir
        num_stages = len(target_spacings) # each stage
        if not isinstance(num_threads, (list, tuple, np.ndarray)):
            num_threads = [num_threads] * num_stages

        assert len(num_threads) == num_stages

        # we need to know which classes are present in this dataset so that we can precompute where these classes are
        # located. This is needed for oversampling foreground
        all_classes_list = load_pickle(join(input_folder_with_cropped_npz, 'dataset_properties.pkl'))['all_classes']  # TODO list

        # 只执行stage1的计划 
        i = num_stages - 1 #2-1
        all_args = []
        output_folder_stage = os.path.join(output_folder, data_identifier + "_stage%d" % i) # 存放resample之后数据的文件目录
        maybe_mkdir_p(output_folder_stage)
        spacing = target_spacings[i]

        for j, case in enumerate(list_of_cropped_npz_files): #TODO
            # case_identifier = case.split("/")[-1][:-4]
            # args = spacing, case_identifier, output_folder_stage, input_folder_with_cropped_npz, force_separate_z, all_classes #all_classes——list
            case_identifier = case # 需要包含train/valid/test 信息
            args = spacing, case_identifier, output_folder_stage, input_folder_with_cropped_npz, force_separate_z, all_classes_list #all_classes——list

            all_args.append(args)

        p = Pool(num_threads[i])
        p.starmap(self._run_internal, all_args)
        p.close()
        p.join()
