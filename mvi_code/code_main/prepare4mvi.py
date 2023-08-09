import SimpleITK as sitk 
import os
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import torch
from multiprocessing import Pool
from collections import OrderedDict
from skimage.transform import resize
from scipy.ndimage import map_coordinates
import copy
import pickle
import csv
import warnings
warnings.filterwarnings("ignore")

def get_uid_to_id_dict():
    data_csv_root_path = './tednet/mvi_code/code_main/final_biomarkers.csv'
    uid_to_id = {}

    with open(data_csv_root_path, 'r') as f:
        reader = csv.reader(f)
        for i in reader:
            if i[0] == 'UID':
                continue
            uid_to_id[i[0]] = i[1]
    return uid_to_id


def write_pickle(obj, file, mode='wb'):
    with open(file, mode) as f:
        pickle.dump(obj, f)

save_pickle = write_pickle

def maybe_mkdir_p(directory):
    directory = os.path.abspath(directory)
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i+1])):
            try:
                os.mkdir(os.path.join("/", *splits[:i+1]))
            except FileExistsError:
                # this can sometimes happen when two jobs try to create the same directory at the same time,
                # especially on network drives.
                print("WARNING: Folder %s already existed and does not need to be created" % directory)

def get_do_separate_z(spacing, anisotropy_threshold=3):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z

def get_lowres_axis(new_spacing):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]
    return axis

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

def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, cval=0, order_z=0):

    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    data = data.astype(float)
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)

    if np.any(shape != new_shape):
        if do_separate_z:
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]): 
                    if axis == 0:
                        reshaped_data.append(resize_fn(data[c, slice_id], new_shape_2d, order, cval=cval, **kwargs))
                    elif axis == 1:
                        reshaped_data.append(resize_fn(data[c, :, slice_id], new_shape_2d, order, cval=cval, **kwargs))
                    else:
                        reshaped_data.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order, cval=cval,
                                                       **kwargs))
                reshaped_data = np.stack(reshaped_data, axis)

                if shape[axis] != new_shape[axis]:
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
                        reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z, cval=cval, 
                                                                   mode='nearest')[None]) 
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                                cval=cval, mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None])
                else:
                    reshaped_final_data.append(reshaped_data[None])
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize_fn(data[c], new_shape, order, cval=cval, **kwargs)[None])
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data

def resample_5mmto1mm(vein_img_path, ref_artery_img_path, vein_roi_path, vein_cap_path, vein_fen_path, each_sample, args):

    maybe_mkdir_p(os.path.join(args.vein_data_path, each_sample))
    vein_img_itk = sitk.ReadImage(vein_img_path)
    vein_img_npy = sitk.GetArrayFromImage(vein_img_itk)
    # vein_img_npy = np.load(vein_img_path)
    venous_spacing = vein_img_itk.GetSpacing()
    original_spacing = venous_spacing

    artery_img_itk = sitk.ReadImage(ref_artery_img_path)
    artery_img_npy = sitk.GetArrayFromImage(artery_img_itk)
    # artery_img_npy = np.load(ref_artery_img_path)
    new_shape = artery_img_npy.shape

    # vein_roi_npy = np.load(vein_roi_path, 'r')
    # vein_cap_npy = np.load(vein_cap_path, 'r')
    # vein_fen_npy = np.load(vein_fen_path, 'r')
    vein_roi_itk = sitk.ReadImage(vein_roi_path)
    vein_roi_npy = sitk.GetArrayFromImage(vein_roi_itk)
    vein_cap_itk = sitk.ReadImage(vein_cap_path)
    vein_cap_npy = sitk.GetArrayFromImage(vein_cap_itk)
    vein_fen_itk = sitk.ReadImage(vein_fen_path)
    vein_fen_npy = sitk.GetArrayFromImage(vein_fen_itk)

    itk_spacing = artery_img_itk.GetSpacing()
    itk_origin = artery_img_itk.GetOrigin()
    itk_direction = artery_img_itk.GetDirection()
    target_spacing = itk_spacing

    max_spacing_axis = np.argmax(original_spacing)
    remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
    transpose_forward = [max_spacing_axis] + remaining_axes
    original_spacing = np.array(original_spacing)[transpose_forward]

    assert vein_img_npy.shape == vein_roi_npy.shape == vein_cap_npy.shape == vein_fen_npy.shape
     
    if len(vein_img_npy.shape) == 3:
        vein_img_npy = vein_img_npy[np.newaxis, :]
        vein_roi_npy = vein_roi_npy[np.newaxis, :]
        vein_cap_npy = vein_cap_npy[np.newaxis, :]
        vein_fen_npy = vein_fen_npy[np.newaxis, :]
        assert len(vein_img_npy.shape) == 4, "seg must be c x y z"

    if get_do_separate_z(original_spacing, 3):
        do_separate_z = True
        axis = get_lowres_axis(original_spacing)
    elif get_do_separate_z(target_spacing, 3):
        do_separate_z = True
        axis = get_lowres_axis(target_spacing)
    else:
        do_separate_z = False
        axis = None

    if axis is not None:
        if len(axis) == 3:
            axis = (0, )
        elif len(axis) == 2:
            print("WARNING: axis has len 2, axis: %s, spacing: %s, target_spacing: %s" % (str(axis), original_spacing, target_spacing))
            do_separate_z = False
        else:
            pass


    img_vein_1mm_npy = resample_data_or_seg(vein_img_npy, new_shape, False, axis, 3, do_separate_z, cval=-1, order_z=0).squeeze()
    roi_vein_1mm_npy = resample_data_or_seg(vein_roi_npy, new_shape, True, axis, 1, do_separate_z, cval=-1, order_z=0).squeeze()
    cap_vein_1mm_npy = resample_data_or_seg(vein_cap_npy, new_shape, True, axis, 1, do_separate_z, cval=-1, order_z=0).squeeze()
    fen_vein_1mm_npy = resample_data_or_seg(vein_fen_npy, new_shape, True, axis, 1, do_separate_z, cval=-1, order_z=0).squeeze()

    assert img_vein_1mm_npy.shape == roi_vein_1mm_npy.shape == cap_vein_1mm_npy.shape == fen_vein_1mm_npy.shape == new_shape

    maybe_mkdir_p(os.path.join(args.vein_data_path, each_sample))

    npy_new = sitk.GetImageFromArray(img_vein_1mm_npy)
    npy_new.SetSpacing(itk_spacing)
    npy_new.SetOrigin(itk_origin)
    npy_new.SetDirection(itk_direction)
    output_nii_path = os.path.join(args.vein_data_path, each_sample, each_sample+'_img_1mm.nii.gz')
    sitk.WriteImage(npy_new, output_nii_path)

    npy_new = sitk.GetImageFromArray(roi_vein_1mm_npy)
    npy_new.SetSpacing(itk_spacing)
    npy_new.SetOrigin(itk_origin)
    npy_new.SetDirection(itk_direction)
    output_nii_path = os.path.join(args.vein_data_path, each_sample, each_sample+'_roi_1mm.nii.gz')
    sitk.WriteImage(npy_new, output_nii_path)

    npy_new = sitk.GetImageFromArray(cap_vein_1mm_npy)
    npy_new.SetSpacing(itk_spacing)
    npy_new.SetOrigin(itk_origin)
    npy_new.SetDirection(itk_direction)
    output_nii_path = os.path.join(args.vein_data_path, each_sample, each_sample+'_cap_1mm.nii.gz')
    sitk.WriteImage(npy_new, output_nii_path)

    npy_new = sitk.GetImageFromArray(fen_vein_1mm_npy)
    npy_new.SetSpacing(itk_spacing)
    npy_new.SetOrigin(itk_origin)
    npy_new.SetDirection(itk_direction)
    output_nii_path = os.path.join(args.vein_data_path, each_sample, each_sample+'_fen_1mm.nii.gz')
    sitk.WriteImage(npy_new, output_nii_path)

def resample_5mmto1mm_multi_pool(args):

    sample_id_list = os.listdir(args.vein_data_path)
    uid_to_id_dict = get_uid_to_id_dict()

    samples_list = []
    for each_sample in sample_id_list:
        if each_sample not in uid_to_id_dict.keys():
            continue
        id_sample = uid_to_id_dict[each_sample]
        
        # vein_img_path = os.path.join(args.vein_data_path, each_sample, each_sample+'_img_roi.npy')
        vein_img_path = os.path.join(args.vein_data_path, each_sample, each_sample+'_img_5mm.nii')
        ref_artery_img_path = os.path.join('./tednet/494_hcc_artery_1mm_ACE', id_sample, 'hcc_surg_artery.nii')

        vein_roi_path = os.path.join(args.vein_data_path, each_sample, each_sample+'_roi_5mm.nii')
        vein_cap_path = os.path.join(args.vein_data_path, each_sample, each_sample+'_cap_5mm.nii')
        vein_fen_path = os.path.join(args.vein_data_path, each_sample, each_sample+'_fen_5mm.nii')

        if not os.path.exists(vein_img_path):
            continue
        if not os.path.exists(ref_artery_img_path):
            continue
        if not os.path.exists(vein_roi_path):
            continue
        if not os.path.exists(vein_cap_path):
            continue
        if not os.path.exists(vein_fen_path):
            continue

        each_sample = [vein_img_path, ref_artery_img_path, vein_roi_path, vein_cap_path, vein_fen_path, each_sample, args]
        samples_list.append(each_sample)

    p = Pool(args.num_thread)
    return_data = p.starmap(resample_5mmto1mm, samples_list)
    p.close()
    p.join()

def get_bbox_3d(roi):
    zmin = np.min(np.nonzero(roi)[0])
    zmax = np.max(np.nonzero(roi)[0])
    rmin = np.min(np.nonzero(roi)[1])
    rmax = np.max(np.nonzero(roi)[1])
    cmin = np.min(np.nonzero(roi)[2])
    cmax = np.max(np.nonzero(roi)[2])
    return zmin, zmax, rmin, rmax, cmin, cmax

def crop4mvi(vein_img_path, vein_roi_path, vein_cap_path, vein_fen_path, artery_img_path, artery_roi_path, artery_tmv_path, artery_ace_path, each_sample, args):

    # each_sample_part_save_path = os.path.join(args.save_path, each_sample, each_sample+'_vein_part.npy')
    each_sample_part_save_path = os.path.join(args.save_path, each_sample, each_sample+'_prepare_mvi.npy')
    each_pkl_path = each_sample_part_save_path[:-4] + '.pkl'

    maybe_mkdir_p(os.path.join(args.save_path, each_sample))
    each_pkl = OrderedDict()

    target_sized = 72
    each_pkl['target_sized'] = target_sized
    patch_size = [target_sized, target_sized, target_sized]

    vein_img_itk = sitk.ReadImage(vein_img_path)
    vein_img_npy = sitk.GetArrayFromImage(vein_img_itk)[None]
    # vein_img_npy = np.load(vein_img_path)[None]
    vein_roi_itk = sitk.ReadImage(vein_roi_path)
    vein_roi_npy = sitk.GetArrayFromImage(vein_roi_itk)[None]
    vein_cap_itk = sitk.ReadImage(vein_cap_path)
    vein_cap_npy = sitk.GetArrayFromImage(vein_cap_itk)[None]
    vein_fen_itk = sitk.ReadImage(vein_fen_path)
    vein_fen_npy = sitk.GetArrayFromImage(vein_fen_itk)[None]

    # artery_img_itk = sitk.ReadImage(artery_img_path)
    # artery_img_npy = sitk.GetArrayFromImage(artery_img_itk)[None]
    artery_img_npy = np.load(artery_img_path)[None]
    # artery_roi_itk = sitk.ReadImage(artery_roi_path)
    # artery_roi_npy = sitk.GetArrayFromImage(artery_roi_itk)[None]
    artery_roi_npy = np.load(artery_roi_path)[None]
    # artery_tmv_itk = sitk.ReadImage(artery_tmv_path)
    # artery_tmv_npy = sitk.GetArrayFromImage(artery_tmv_itk)[None]
    if os.path.exists(artery_tmv_path):
        artery_tmv_npy = np.load(artery_tmv_path)[None]
    else:
        artery_tmv_npy = np.zeros_like(artery_roi_npy)
    # artery_ace_itk = sitk.ReadImage(artery_ace_path)
    # artery_ace_npy = sitk.GetArrayFromImage(artery_ace_itk)[None]
    if os.path.exists(artery_ace_path):
        artery_ace_npy = np.load(artery_ace_path)[None]
    else:
        artery_ace_npy = np.zeros_like(artery_roi_npy)

    assert vein_img_npy.shape == vein_roi_npy.shape == vein_cap_npy.shape == vein_fen_npy.shape == artery_img_npy.shape == artery_roi_npy.shape == artery_tmv_npy.shape == artery_ace_npy.shape
    assert len(vein_img_npy.shape) == 4

    if np.sum(artery_ace_npy) > 0:
        each_pkl['have_ace'] = 1
    else:
        each_pkl['have_ace'] = 0
        
    if np.sum(artery_tmv_npy) > 0:
        each_pkl['have_artery'] = 1
    else:
        each_pkl['have_artery'] = 0

    each_pkl['only_id'] = str(each_sample)

    img_shape = vein_img_npy.shape[1:]
    each_pkl['img_shape'] = img_shape
    plus_roi = copy.deepcopy(vein_roi_npy)

    # plus_roi = ace + plus_roi
    plus_roi[plus_roi!= 0]=1 

    zmin, zmax, rmin, rmax, cmin, cmax = get_bbox_3d(plus_roi[0])
    each_pkl['ori_bbox'] = [zmin, zmax, rmin, rmax, cmin, cmax]
    center_z = (zmin+zmax)//2
    center_r = (rmin+rmax)//2
    center_c = (cmin+cmax)//2
    center_point = [center_z, center_r, center_c]
    each_pkl['ori_center_point'] = center_point
    rect_size = [zmax-zmin, rmax-rmin, cmax-cmin]
    each_pkl['ori_rect_size'] = rect_size
    # ori_max_side = np.max(rect_size)

    # 还要保证足够的上下文，对roi的尺寸进行扩大
    valid_size_z = rect_size[0]+24
    valid_size_r = rect_size[1]+24
    valid_size_c = rect_size[2]+24
    valid_size = [valid_size_z, valid_size_r, valid_size_c]

    # 我们只考虑90度的rotation情况
    crop_size_plus = valid_size

    # 转为正方体
    crop_size_plus_max = np.max(crop_size_plus)
    crop_size_plus_square = np.array([crop_size_plus_max,crop_size_plus_max,crop_size_plus_max])
    crop_size_plus = crop_size_plus_square
    each_pkl['final_rect_size'] = crop_size_plus

    resize_factor = crop_size_plus[0]/ target_sized
    each_pkl['resize_factor'] = resize_factor
    # resize_factor_list.append(resize_factor)

    final_selected_points = center_point

    # 若原图的尺寸不足以直接进行crop
    z1_for_crop = min(final_selected_points[0] - crop_size_plus[0]//2, 0)
    z2_for_crop = max(final_selected_points[0] + crop_size_plus[0]//2 + crop_size_plus[0] % 2,img_shape[0]-1)

    r1_for_crop = min(final_selected_points[1] - crop_size_plus[1]//2,0)
    r2_for_crop = max(final_selected_points[1] + crop_size_plus[1]//2 + crop_size_plus[1] % 2,img_shape[1]-1)

    c1_for_crop = min(final_selected_points[2] - crop_size_plus[2]//2,0)
    c2_for_crop = max(final_selected_points[2] + crop_size_plus[2]//2 + crop_size_plus[2] % 2,img_shape[2]-1)

    # 根据中心点和边长 估算原图padding后的尺寸
    each_pkl['ref_padding'] = [z1_for_crop, z2_for_crop, r1_for_crop, r2_for_crop, c1_for_crop, c2_for_crop]
    ori_img_for_crop = [z2_for_crop-z1_for_crop, r2_for_crop-r1_for_crop, c2_for_crop-c1_for_crop]
    each_pkl['ori_img_for_crop'] = ori_img_for_crop

    
    data_all = vein_img_npy
    data_all = np.concatenate((data_all, artery_img_npy), axis=0)
    seg_all = np.concatenate((vein_roi_npy, vein_cap_npy, vein_fen_npy), axis=0)
    seg_all = np.concatenate((seg_all, artery_roi_npy, artery_tmv_npy, artery_ace_npy), axis=0)

    data_all_after_padding = np.pad(data_all, ((0, 0), 
                    (-z1_for_crop, z2_for_crop-img_shape[0]+1),
                    (-r1_for_crop, r2_for_crop-img_shape[1]+1),
                    (-c1_for_crop, c2_for_crop-img_shape[2]+1)),
                    "constant", **OrderedDict())

    seg_all_after_padding = np.pad(seg_all, ((0, 0), 
                    (-z1_for_crop, z2_for_crop-img_shape[0]+1),
                    (-r1_for_crop, r2_for_crop-img_shape[1]+1),
                    (-c1_for_crop, c2_for_crop-img_shape[2]+1)),
                    "constant", **{'constant_values': -1})

    # crop, ()中为cropped后的新中心点坐标
    valid_bbox_x_lb = (final_selected_points[0] - z1_for_crop) - crop_size_plus[0]//2
    valid_bbox_x_ub = (final_selected_points[0] - z1_for_crop) + crop_size_plus[0]//2 + crop_size_plus[0] % 2
    valid_bbox_y_lb = (final_selected_points[1] - r1_for_crop) - crop_size_plus[1]//2
    valid_bbox_y_ub = (final_selected_points[1] - r1_for_crop) + crop_size_plus[1]//2 + crop_size_plus[1] % 2
    valid_bbox_z_lb = (final_selected_points[2] - c1_for_crop) - crop_size_plus[2]//2
    valid_bbox_z_ub = (final_selected_points[2] - c1_for_crop) + crop_size_plus[2]//2 + crop_size_plus[2] % 2

    each_pkl['final_crop'] = [valid_bbox_x_lb,valid_bbox_x_ub,valid_bbox_y_lb,valid_bbox_y_ub,valid_bbox_z_lb,valid_bbox_z_ub]

    case_all_data_cropped = np.copy(data_all_after_padding[:, valid_bbox_x_lb:valid_bbox_x_ub, valid_bbox_y_lb:valid_bbox_y_ub, valid_bbox_z_lb:valid_bbox_z_ub])
    case_all_seg_cropped = np.copy(seg_all_after_padding[:, valid_bbox_x_lb:valid_bbox_x_ub, valid_bbox_y_lb:valid_bbox_y_ub, valid_bbox_z_lb:valid_bbox_z_ub])

    vein_data_after_resized = resize(case_all_data_cropped[0], patch_size, order=3, cval=0, **{'mode': 'edge', 'anti_aliasing': False})
    artery_data_after_resized = resize(case_all_data_cropped[1], patch_size, order=3, cval=0, **{'mode': 'edge', 'anti_aliasing': False})

    vein_roi_after_resized = resize_segmentation(case_all_seg_cropped[0], patch_size, order=1, cval=-1, **OrderedDict())
    vein_cap_after_resized = resize_segmentation(case_all_seg_cropped[1], patch_size, order=1, cval=-1, **OrderedDict())
    vein_fen_after_resized = resize_segmentation(case_all_seg_cropped[2], patch_size, order=1, cval=-1, **OrderedDict())
    artery_roi_after_resized = resize_segmentation(case_all_seg_cropped[3], patch_size, order=1, cval=-1, **OrderedDict())
    artery_tmv_after_resized = resize_segmentation(case_all_seg_cropped[4], patch_size, order=1, cval=-1, **OrderedDict())
    artery_ace_after_resized = resize_segmentation(case_all_seg_cropped[5], patch_size, order=1, cval=-1, **OrderedDict())

    vein_data_after_resized = vein_data_after_resized[None][None]
    artery_data_after_resized = artery_data_after_resized[None][None]

    vein_roi_after_resized = vein_roi_after_resized[None][None]
    vein_roi_after_resized[vein_roi_after_resized == -1] = 0
    vein_cap_after_resized = vein_cap_after_resized[None][None]
    vein_cap_after_resized[vein_cap_after_resized == -1] = 0
    vein_fen_after_resized = vein_fen_after_resized[None][None]
    vein_fen_after_resized[vein_fen_after_resized == -1] = 0
    artery_roi_after_resized = artery_roi_after_resized[None][None]
    artery_roi_after_resized[artery_roi_after_resized == -1] = 0
    artery_tmv_after_resized = artery_tmv_after_resized[None][None]
    artery_tmv_after_resized[artery_tmv_after_resized == -1] = 0
    artery_ace_after_resized = artery_ace_after_resized[None][None]
    artery_ace_after_resized[artery_ace_after_resized == -1] = 0


    each_data_shape = (1, 8, *patch_size)
    each_data = np.zeros(each_data_shape, dtype=np.float32) 
    each_data[0, 0:1] = vein_data_after_resized
    each_data[0, 1:2] = artery_data_after_resized
    each_data[0, 2:3] = vein_roi_after_resized
    each_data[0, 3:4] = vein_cap_after_resized
    each_data[0, 4:5] = vein_fen_after_resized
    each_data[0, 5:6] = artery_roi_after_resized
    each_data[0, 6:7] = artery_tmv_after_resized
    each_data[0, 7:8] = artery_ace_after_resized

    np.save(each_sample_part_save_path, each_data.astype(np.float32))
    save_pickle(each_pkl, each_pkl_path)
    
    return each_sample_part_save_path, each_pkl_path

def crop4mvi_multi_pool(args):

    sample_id_list = os.listdir(args.vein_data_path)
    uid_to_id_dict = get_uid_to_id_dict()

    samples_list = []
    for each_sample in sample_id_list:
        if each_sample not in uid_to_id_dict.keys():
            continue
        id_sample = uid_to_id_dict[each_sample]

        vein_img_path = os.path.join(args.vein_data_path, each_sample, each_sample+'_img_1mm.nii.gz')
        vein_roi_path = os.path.join(args.vein_data_path, each_sample, each_sample+'_roi_1mm.nii.gz')
        vein_cap_path = os.path.join(args.vein_data_path, each_sample, each_sample+'_cap_1mm.nii.gz')
        vein_fen_path = os.path.join(args.vein_data_path, each_sample, each_sample+'_fen_1mm.nii.gz')

        artery_img_path = os.path.join(args.artery_data_path, id_sample, id_sample+'_img_roi.npy')
        artery_roi_path = os.path.join(args.artery_data_path, id_sample, id_sample+'_roi.npy')
        artery_tmv_path = os.path.join(args.artery_data_path, id_sample, id_sample+'_tmv.npy')
        artery_ace_path = os.path.join(args.artery_data_path, id_sample, id_sample+'_ace.npy')

        if not os.path.exists(vein_img_path):
            continue
        if not os.path.exists(vein_roi_path):
            continue
        if not os.path.exists(vein_cap_path):
            continue
        if not os.path.exists(vein_fen_path):
            continue

        if not os.path.exists(artery_img_path):
            continue
        if not os.path.exists(artery_roi_path):
            continue
        # if not os.path.exists(artery_tmv_path):
        #     continue
        # if not os.path.exists(artery_ace_path):
        #     continue

        each_sample = [vein_img_path, vein_roi_path, vein_cap_path, vein_fen_path, artery_img_path, artery_roi_path, artery_tmv_path, artery_ace_path, id_sample, args]
        samples_list.append(each_sample)

    p = Pool(args.num_thread)
    return_data = p.starmap(crop4mvi, samples_list)
    p.close()
    p.join()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--vein_data_path', type=str, default='', help='raw_data_path') # vein
    parser.add_argument('--artery_data_path', type=str, default='', help='raw_data_path') # artery
    parser.add_argument('--save_path', type=str, default='', help='where to save')
    parser.add_argument('--num_thread', type=int, default=1, help='multi-process')
    parser.add_argument('--show', action='store_true', help='vis')

    args = parser.parse_args()

    # resample_5mmto1mm_multi_pool(args)
    crop4mvi_multi_pool(args)
    print('prepare4mvi finish!')
