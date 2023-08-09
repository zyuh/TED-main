import numpy as np
from utils.file_ops import *
import copy
from collections import OrderedDict
import csv
from multiprocessing import Pool
from skimage.transform import resize
import SimpleITK as sitk

from preprocess.preprocessor import resample_data_or_seg


def get_do_separate_z(spacing, anisotropy_threshold=3):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z

def get_lowres_axis(new_spacing):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]
    return axis

def write_pickle(obj, file, mode='wb'):
    with open(file, mode) as f:
        pickle.dump(obj, f)

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

def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))

    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)

    from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)

    if len(coords) == 3:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0) # 0是求max的维度
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)

    final_shape /= min(scale_range)
    return final_shape.astype(int)

def get_rotation_size():
    patch_size_plus = [64,64,64]
    rotation_x = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
    rotation_y = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
    rotation_z = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
    basic_generator_patch_size = get_patch_size(patch_size_plus, rotation_x, rotation_y, rotation_z, (0.85, 1.25))
    print(basic_generator_patch_size)

def get_bbox_3d(roi): # 根据roi得到bbox
    zmin = np.min(np.nonzero(roi)[0])
    zmax = np.max(np.nonzero(roi)[0])
    rmin = np.min(np.nonzero(roi)[1])
    rmax = np.max(np.nonzero(roi)[1])
    cmin = np.min(np.nonzero(roi)[2])
    cmax = np.max(np.nonzero(roi)[2])
    return zmin, zmax, rmin, rmax, cmin, cmax

def get_xyz_roi_range():
    root_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/preprocessed_data/nnUNetData_plans_v2.1_stage1/labels_roi'
    train_roi_list = subfiles(join(root_path,'train'), join=True, suffix='.npy')
    valid_roi_list = subfiles(join(root_path,'valid'), join=True, suffix='.npy')
    test_roi_list = subfiles(join(root_path,'test'), join=True, suffix='.npy')
    total_list = train_roi_list + valid_roi_list + test_roi_list
    z_range_list = []
    r_range_list = []
    c_range_list = []
    print(len(total_list))
    for each_sample_path in total_list:
        each_sample_npy = np.load(each_sample_path, 'r')
        each_sample_npy = each_sample_npy.squeeze()
        only_id = each_sample_path.split('/')[-1][:-4]
        zmin, zmax, rmin, rmax, cmin, cmax = get_bbox_3d(each_sample_npy)
        z_range = zmax - zmin
        r_range = rmax - rmin
        c_range = cmax - cmin
        print(only_id, z_range, r_range, c_range)
        z_range_list.append(z_range)
        r_range_list.append(r_range)
        c_range_list.append(c_range)
    print('='*20)
    print(np.max(z_range_list), np.min(z_range_list), np.mean(z_range_list))
    print(np.max(r_range_list), np.min(r_range_list), np.mean(r_range_list))
    print(np.max(c_range_list), np.min(c_range_list), np.mean(c_range_list))


'''labels_artery_new (pred) | labels_artery (ori gt)'''
def prepare_dataV2(split_flag='train', normalization_flag='roi_cap'):

    root_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/preprocessed_data/nnUNetData_plans_v2.1_stage1'
    if split_flag=='train':
        data_path = join(root_path, 'imagesTr')
        venous_data_path = join(root_path, 'imagesTr_venous')
        roi_path = join(root_path, 'labels_roi', 'train')
        artery_path = join(root_path, 'labels_artery', 'train')
        ace_path = join(root_path, 'labels_ACE', 'train')
        fen_path = join(root_path, 'labels_fen', 'train')
        cap_path = join(root_path, 'labels_cap', 'train')
    elif split_flag=='valid':
        data_path = join(root_path, 'imagesTv')
        venous_data_path = join(root_path, 'imagesTv_venous')
        roi_path = join(root_path, 'labels_roi', 'valid')
        artery_path = join(root_path, 'labels_artery', 'valid')
        ace_path = join(root_path, 'labels_ACE', 'valid')
        fen_path = join(root_path, 'labels_fen', 'valid')
        cap_path = join(root_path, 'labels_cap', 'valid')
    elif split_flag=='test':
        data_path = join(root_path, 'imagesTs')
        venous_data_path = join(root_path, 'imagesTs_venous')
        roi_path = join(root_path, 'labels_roi', 'test') 
        artery_path = join(root_path, 'labels_artery', 'test')
        ace_path = join(root_path, 'labels_ACE', 'test')
        fen_path = join(root_path, 'labels_fen', 'test')
        cap_path = join(root_path, 'labels_cap', 'test')
    else:
        raise ValueError
    
    if normalization_flag=='roi_roi':
        data_path = join(data_path, '_roi')  
        venous_data_path = join(venous_data_path, '_roi')  
    elif normalization_flag=='artery_roi':
        data_path = join(data_path, '_artery')  
        venous_data_path = join(venous_data_path, '_roi')  
    elif normalization_flag=='roi_cap':
        data_path = join(data_path, '_roi')  
        venous_data_path = join(venous_data_path, '_cap')  
    elif normalization_flag=='artery_cap':
        data_path = join(data_path, '_artery')  
        venous_data_path = join(venous_data_path, '_cap')  
    else:
        raise ValueError

    lists = []
    patients = subfiles(data_path, join=True, suffix='.npy')
    if split_flag=='train':
        assert len(patients) == 331
    elif split_flag=='valid':
        assert len(patients) == 49
    elif split_flag=='test':
        assert len(patients) == 97
    else:
        raise ValueError
    
    patients.sort()

    csv_path = '/GPFS/data/yuhangzhou/yuhangzhou/nnunet_dataset/nnUNet_raw/nnUNet_raw_data/Task300_liverMVI/log_loader.csv'
    id_to_mvi = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for each_line in reader:
            if each_line[1] == 'UID': # 跳过表头
                continue
            id_to_mvi[each_line[0]] = each_line[-1] #id 不带后缀
    f.close()

    for th in range(len(patients)):
        patient = patients[th]
        case_id = patient.split('/')[-1]
        only_id = case_id[:-4]

        data = join(data_path, case_id)
        venous_data = join(venous_data_path, case_id)
        roi = join(roi_path, case_id)
        artery = join(artery_path, case_id)
        ace = join(ace_path, case_id)
        cap = join(cap_path, case_id)
        fen = join(fen_path, case_id)

        mvi = id_to_mvi[only_id]
        mvi = int(mvi)
        assert mvi==0 or mvi==1

        each_patient_data_list = [data, venous_data, roi, artery, ace, cap, fen, mvi, only_id]
        lists.append(each_patient_data_list)
    
    return lists



def show3d_each_sample_in_nii(case_id, split_flag='test', normalization_flag='roi_cap'):
    # data artery ace roi venous_data cap fen # each_data[0, 0:1]
    # only_id = str(1011402)
    only_id = case_id
    root_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/preprocessed_data/nnUNetData_plans_v2.1_stage1/resized_data_gt'
    root_path = join(root_path, split_flag)

    each_data = join(root_path, only_id + '.npy')
    each_pkl = join(root_path, only_id + '.pkl')

    each_data_npy = np.load(each_data, 'r')
    data = each_data_npy[0,0]
    artery = each_data_npy[0,1]
    ace = each_data_npy[0,2]
    roi = each_data_npy[0,3]
    venous_data = each_data_npy[0,4]
    cap = each_data_npy[0,5]
    fen = each_data_npy[0,6]
    
    # data 部分要做反标准化
    venous_plan_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/preprocessed_data/nnUNetPlansv2.1_plans_3D_venous.pkl'
    artery_plan_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/preprocessed_data/nnUNetPlansv2.1_plans_3D.pkl'
    venous_plan = load_pickle(venous_plan_path)
    artery_plan = load_pickle(artery_plan_path)

    # *std+mean
    intensityproperties_venous = venous_plan['dataset_properties']['intensityproperties']
    mean_intensity_venous_cap = intensityproperties_venous[0]['mean_cap']
    std_intensity_venous_cap = intensityproperties_venous[0]['sd_cap']
    mean_intensity_venous_roi = intensityproperties_venous[0]['mean_roi']
    std_intensity_venous_roi = intensityproperties_venous[0]['sd_roi']


    intensityproperties_artery = artery_plan['dataset_properties']['intensityproperties']
    mean_intensity_artery_artery = intensityproperties_artery[0]['mean_artery']
    std_intensity_artery_artery = intensityproperties_artery[0]['sd_artery']
    mean_intensity_artery_roi = intensityproperties_artery[0]['mean_roi']
    std_intensity_artery_roi = intensityproperties_artery[0]['sd_roi']

    if normalization_flag == 'roi_cap':
        artery_data = data * std_intensity_artery_roi + mean_intensity_artery_roi
        # venous_data_roi = venous_data * std_intensity_venous_roi + mean_intensity_venous_roi
        # artery_data_artery = data * std_intensity_artery_artery + mean_intensity_artery_artery
        venous_data = venous_data * std_intensity_venous_cap + mean_intensity_venous_cap


    # 然后转换为nii存储
    save_root_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/preprocessed_data/nnUNetData_plans_v2.1_stage1/error_show'
    case_root_path = join(save_root_path, only_id)
    maybe_mkdir_p(case_root_path)


    if split_flag=='train':
        ref_path = '/GPFS/data/yuhangzhou/yuhangzhou/nnunet_dataset/nnUNet_raw/nnUNet_raw_data/Task300_liverMVI/imagesTr/'+ only_id +'_0000.nii.gz'
    elif split_flag=='valid':
        ref_path = '/GPFS/data/yuhangzhou/yuhangzhou/nnunet_dataset/nnUNet_raw/nnUNet_raw_data/Task300_liverMVI/imagesTv/'+ only_id +'_0000.nii.gz'
    elif split_flag=='test':
        ref_path = '/GPFS/data/yuhangzhou/yuhangzhou/nnunet_dataset/nnUNet_raw/nnUNet_raw_data/Task300_liverMVI/imagesTs/'+ only_id +'_0000.nii.gz'
    else:
        raise ValueError

    ref = sitk.ReadImage(ref_path)
    itk_spacing = ref.GetSpacing()
    itk_origin = ref.GetOrigin()
    itk_direction = ref.GetDirection()

    npy_list = [artery_data, venous_data, ace, artery, cap, fen, roi]
    name_list = ['artery_data', 'venous_data', 'ace', 'artery', 'cap', 'fen', 'roi']

    for th in range(len(name_list)):
        npy_seg_new = npy_list[th]
        if np.sum(npy_seg_new) == 0:
            continue
        name_to_save = name_list[th]
        output_nii_path = join(case_root_path, name_to_save+'.nii.gz')

        npy_seg = sitk.GetImageFromArray(npy_seg_new)
        npy_seg.SetSpacing(itk_spacing)
        npy_seg.SetOrigin(itk_origin)
        npy_seg.SetDirection(itk_direction)
        sitk.WriteImage(npy_seg, output_nii_path)




def show3d_each_sample_in_nii_pred(case_id, split_flag='test', normalization_flag='roi_cap'):
    # data artery ace roi venous_data cap fen # each_data[0, 0:1]
    only_id = case_id
    root_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/preprocessed_data/nnUNetData_plans_v2.1_stage1/resized_data_pred'
    root_path = join(root_path, split_flag)
    each_data = join(root_path, only_id + '.npy')

    each_data_npy = np.load(each_data, 'r')
    label_cap = each_data_npy[0,0]
    label_cap_semantic = each_data_npy[0,1]
    label_fen = each_data_npy[0,2]
    label_fen_semantic = each_data_npy[0,3]
    pred_cap = each_data_npy[0,4]
    new_artery = each_data_npy[0,5]
    old_artery = each_data_npy[0,6]

    # 然后转换为nii存储
    save_root_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/preprocessed_data/nnUNetData_plans_v2.1_stage1/error_show'
    case_root_path = join(save_root_path, only_id)
    maybe_mkdir_p(case_root_path)

    if split_flag=='train':
        ref_path = '/GPFS/data/yuhangzhou/yuhangzhou/nnunet_dataset/nnUNet_raw/nnUNet_raw_data/Task300_liverMVI/imagesTr/'+ only_id +'_0000.nii.gz'
    elif split_flag=='valid':
        ref_path = '/GPFS/data/yuhangzhou/yuhangzhou/nnunet_dataset/nnUNet_raw/nnUNet_raw_data/Task300_liverMVI/imagesTv/'+ only_id +'_0000.nii.gz'
    elif split_flag=='test':
        ref_path = '/GPFS/data/yuhangzhou/yuhangzhou/nnunet_dataset/nnUNet_raw/nnUNet_raw_data/Task300_liverMVI/imagesTs/'+ only_id +'_0000.nii.gz'
    else:
        raise ValueError

    ref = sitk.ReadImage(ref_path)
    itk_spacing = ref.GetSpacing()
    itk_origin = ref.GetOrigin()
    itk_direction = ref.GetDirection()

    npy_list = [label_cap, label_cap_semantic, label_fen, label_fen_semantic, pred_cap, new_artery, old_artery]
    name_list = ['label_cap', 'label_cap_semantic', 'label_fen', 'label_fen_semantic', 'pred_cap', 'new_artery', 'old_artery']

    for th in range(len(name_list)):
        npy_seg_new = npy_list[th]
        if np.sum(npy_seg_new) == 0:
            continue
        name_to_save = name_list[th]
        output_nii_path = join(case_root_path, name_to_save+'.nii.gz')

        npy_seg = sitk.GetImageFromArray(npy_seg_new)
        npy_seg.SetSpacing(itk_spacing)
        npy_seg.SetOrigin(itk_origin)
        npy_seg.SetDirection(itk_direction)
        sitk.WriteImage(npy_seg, output_nii_path)


def each_sample_generate(data_path, venous_data_path, roi_path, artery_path, ace_path, 
        cap_path, fen_path, mvi_value, only_id, phase, save_root_path):
    print(only_id)

    each_sample_fusion_save_path = join(save_root_path, 'resized_data_gt', phase, only_id+'.npy')
    each_pkl_path = each_sample_fusion_save_path[:-4] + '.pkl'
    maybe_mkdir_p(join(save_root_path, 'resized_data_gt', phase))
    each_pkl = OrderedDict()

    # NOTE：留出随机crop的空间，不是64, 所有都resize 到72
    target_sized = 72
    each_pkl['target_sized'] = target_sized
    patch_size = [target_sized, target_sized, target_sized]
    # NOTE：记得保存resize scale 以及 mvi 以及bbox

    img = np.load(data_path, 'r')
    venous_img = np.load(venous_data_path, 'r')
    roi = np.load(roi_path, 'r')
    if os.path.exists(ace_path):
        ace = np.load(ace_path, 'r')
        each_pkl['have_ace'] = 1
    else:
        ace = np.zeros_like(img)
        each_pkl['have_ace'] = 0
        
    if os.path.exists(artery_path):
        artery = np.load(artery_path, 'r')
        each_pkl['have_artery'] = 1
    else:
        artery = np.zeros_like(img)
        each_pkl['have_artery'] = 0
    cap = np.load(cap_path, 'r')
    fen = np.load(fen_path, 'r')

    only_id = str(only_id)
    mvi = int(mvi_value)

    each_pkl['only_id'] = only_id
    each_pkl['mvi'] = mvi
    # ================================================ #

    img_shape = img.shape[1:]
    each_pkl['img_shape'] = img_shape
    plus_roi = copy.deepcopy(roi) #mean~64 max~224 min~10
    # if np.sum(ace) != 0:
    #     scale_size = 1.25
    # else:
    #     scale_size = 1.5 # 没有ace的把范围扩大一点
    plus_roi = ace + plus_roi # ace 可能全0
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
    ori_max_side = np.max(rect_size)

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

    data_all = np.concatenate((img, venous_img), axis=0)
    seg_all = np.concatenate((roi, artery, ace, cap, fen), axis=0)

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

    data_after_resized = resize(case_all_data_cropped[0], patch_size, order=3, cval=0, **{'mode': 'edge', 'anti_aliasing': False})
    venous_data_after_resized = resize(case_all_data_cropped[1], patch_size, order=3, cval=0, **{'mode': 'edge', 'anti_aliasing': False})

    roi_after_resized = resize_segmentation(case_all_seg_cropped[0], patch_size, order=1, cval=-1, **OrderedDict())
    artery_after_resized = resize_segmentation(case_all_seg_cropped[1], patch_size, order=1, cval=-1, **OrderedDict())
    ace_after_resized = resize_segmentation(case_all_seg_cropped[2], patch_size, order=1, cval=-1, **OrderedDict())
    cap_after_resized = resize_segmentation(case_all_seg_cropped[3], patch_size, order=1, cval=-1, **OrderedDict())
    fen_after_resized = resize_segmentation(case_all_seg_cropped[4], patch_size, order=1, cval=-1, **OrderedDict())

    if phase=='train':
        pass
        # axis = np.random.randint(0, 3)
        # axis = 0
        # data_after_resized = np.flip(data_after_resized, axis=axis)
        # venous_data_after_resized = np.flip(venous_data_after_resized, axis=axis)
        # roi_after_resized = np.flip(roi_after_resized, axis=axis)
        # artery_after_resized = np.flip(artery_after_resized, axis=axis)
        # ace_after_resized = np.flip(ace_after_resized, axis=axis)
        # cap_after_resized = np.flip(cap_after_resized, axis=axis)
        # fen_after_resized = np.flip(fen_after_resized, axis=axis)

        # if do_rotation:
        # k = np.random.randint(0, 4)
        # k = 1
        # rot_axis_list = [(0,1),(1,2),(0,2)]
        # rot_axis_th = np.random.randint(0, 3)
        # rot_axis_th = 0
        # rot_axis = rot_axis_list[rot_axis_th]
        # data_after_resized = np.rot90(data_after_resized, k, rot_axis)
        # venous_data_after_resized = np.rot90(venous_data_after_resized, k, rot_axis)
        # roi_after_resized = np.rot90(roi_after_resized, k, rot_axis)
        # artery_after_resized = np.rot90(artery_after_resized, k, rot_axis)
        # ace_after_resized = np.rot90(ace_after_resized, k, rot_axis)
        # cap_after_resized = np.rot90(cap_after_resized, k, rot_axis)
        # fen_after_resized = np.rot90(fen_after_resized, k, rot_axis)

    data_after_resized = data_after_resized[None][None]
    venous_data_after_resized = venous_data_after_resized[None][None]
    roi_after_resized = roi_after_resized[None][None]
    roi_after_resized[roi_after_resized == -1] = 0
    artery_after_resized = artery_after_resized[None][None]
    ace_after_resized = ace_after_resized[None][None]
    cap_after_resized = cap_after_resized[None][None]
    fen_after_resized = fen_after_resized[None][None]
    artery_after_resized[artery_after_resized == -1] = 0
    ace_after_resized[ace_after_resized == -1] = 0
    cap_after_resized[cap_after_resized == -1] = 0
    fen_after_resized[fen_after_resized == -1] = 0

    # patch_size = [target_sized, target_sized, target_sized]
    each_data_shape = (1, 7, *patch_size)
    each_data = np.zeros(each_data_shape, dtype=np.float32) 
    each_data[0, 0:1] = data_after_resized
    each_data[0, 1:2] = artery_after_resized
    each_data[0, 2:3] = ace_after_resized
    each_data[0, 3:4] = roi_after_resized
    each_data[0, 4:5] = venous_data_after_resized
    each_data[0, 5:6] = cap_after_resized
    each_data[0, 6:7] = fen_after_resized

    # print('saving npy...', each_data.shape)
    np.save(each_sample_fusion_save_path, each_data.astype(np.float32))
    # print('saving...', each_pkl)
    # write_pickle(each_pkl, each_pkl_path)
    save_pickle(each_pkl, each_pkl_path)
    
    return each_sample_fusion_save_path, each_pkl_path


def each_sample_generate_pred(label_cap_path, label_cap_semantic_path, pred_cap_path, 
    label_fen_path, label_fen_semantic_path, labels_artery_new_path, labels_artery_old_path, each_id, save_npy_path, save_pkl_path):
    print(each_id, save_npy_path.split('/')[-2])

    ref_pkl_path = save_pkl_path.replace('resized_data_pred', 'resized_data_gt')
    ref_pkl = load_pickle(ref_pkl_path)

    each_sample_fusion_save_path = save_npy_path
    each_pkl_path = save_pkl_path
    each_pkl = OrderedDict()

    # NOTE：留出随机crop的空间，不是64, 所有都resize 到72
    target_sized = 72
    each_pkl['target_sized'] = target_sized
    patch_size = [target_sized, target_sized, target_sized]
    # NOTE：记得保存resize scale 以及 mvi 以及bbox

    label_cap = sitk.GetArrayFromImage(sitk.ReadImage(label_cap_path))[None]
    label_cap_semantic = sitk.GetArrayFromImage(sitk.ReadImage(label_cap_semantic_path))[None]
    label_fen = sitk.GetArrayFromImage(sitk.ReadImage(label_fen_path))[None]
    label_fen_semantic = sitk.GetArrayFromImage(sitk.ReadImage(label_fen_semantic_path))[None]
    pred_cap = sitk.GetArrayFromImage(sitk.ReadImage(pred_cap_path))[None]
   
    if os.path.exists(labels_artery_new_path):
        assert os.path.exists(labels_artery_old_path)
        new_artery = np.load(labels_artery_new_path, 'r')
        if each_id == '2285036':
            old_artery = sitk.GetArrayFromImage(sitk.ReadImage(labels_artery_old_path))[:-1][None]
        else:
            old_artery = sitk.GetArrayFromImage(sitk.ReadImage(labels_artery_old_path))[:-2][None]
    else:
        assert not os.path.exists(labels_artery_old_path)
        new_artery = np.zeros_like(label_cap)
        old_artery = np.zeros_like(label_cap)

    assert label_cap.shape==label_cap_semantic.shape==label_fen.shape==label_fen_semantic.shape==pred_cap.shape
    # print(label_cap_semantic.shape, new_artery.shape, old_artery.shape) # 3维 4维
    # ================================================ #
    img_shape = ref_pkl['img_shape']
    # print(img_shape)
    [zmin, zmax, rmin, rmax, cmin, cmax] = ref_pkl['ori_bbox']
    center_point = ref_pkl['ori_center_point']
    rect_size = ref_pkl['ori_rect_size']
    crop_size_plus = ref_pkl['final_rect_size']
    resize_factor = ref_pkl['resize_factor']
    [z1_for_crop, z2_for_crop, r1_for_crop, r2_for_crop, c1_for_crop, c2_for_crop] = ref_pkl['ref_padding']
    ori_img_for_crop = ref_pkl['ori_img_for_crop']

    seg_all = np.concatenate((label_cap, label_cap_semantic, label_fen, label_fen_semantic, pred_cap, new_artery, old_artery), axis=0)

    seg_all_after_padding = np.pad(seg_all, ((0, 0), 
                    (-z1_for_crop, z2_for_crop-img_shape[0]+1),
                    (-r1_for_crop, r2_for_crop-img_shape[1]+1),
                    (-c1_for_crop, c2_for_crop-img_shape[2]+1)),
                    "constant", **{'constant_values': -1})

    [valid_bbox_x_lb,valid_bbox_x_ub,valid_bbox_y_lb,valid_bbox_y_ub,valid_bbox_z_lb,valid_bbox_z_ub] = ref_pkl['final_crop']

    case_all_seg_cropped = np.copy(seg_all_after_padding[:, valid_bbox_x_lb:valid_bbox_x_ub, valid_bbox_y_lb:valid_bbox_y_ub, valid_bbox_z_lb:valid_bbox_z_ub])

    label_cap_after_resized = resize_segmentation(case_all_seg_cropped[0], patch_size, order=1, cval=-1, **OrderedDict())
    label_cap_semantic_after_resized = resize_segmentation(case_all_seg_cropped[1], patch_size, order=1, cval=-1, **OrderedDict())
    label_fen_after_resized = resize_segmentation(case_all_seg_cropped[2], patch_size, order=1, cval=-1, **OrderedDict())
    label_fen_semantic_after_resized = resize_segmentation(case_all_seg_cropped[3], patch_size, order=1, cval=-1, **OrderedDict())
    pred_cap_after_resized = resize_segmentation(case_all_seg_cropped[4], patch_size, order=1, cval=-1, **OrderedDict())
    new_artery_after_resized = resize_segmentation(case_all_seg_cropped[5], patch_size, order=1, cval=-1, **OrderedDict())
    old_artery_after_resized = resize_segmentation(case_all_seg_cropped[6], patch_size, order=1, cval=-1, **OrderedDict())


    label_cap_after_resized = label_cap_after_resized[None][None]
    label_cap_semantic_after_resized = label_cap_semantic_after_resized[None][None]
    label_fen_after_resized = label_fen_after_resized[None][None]
    label_fen_semantic_after_resized = label_fen_semantic_after_resized[None][None]
    pred_cap_after_resized = pred_cap_after_resized[None][None]
    new_artery_after_resized = new_artery_after_resized[None][None]
    old_artery_after_resized = old_artery_after_resized[None][None]

    label_cap_after_resized[label_cap_after_resized == -1] = 0
    label_cap_semantic_after_resized[label_cap_semantic_after_resized == -1] = 0
    label_fen_after_resized[label_fen_after_resized == -1] = 0
    label_fen_semantic_after_resized[label_fen_semantic_after_resized == -1] = 0
    pred_cap_after_resized[pred_cap_after_resized == -1] = 0
    new_artery_after_resized[new_artery_after_resized == -1] = 0
    old_artery_after_resized[old_artery_after_resized == -1] = 0

    # patch_size = [target_sized, target_sized, target_sized]
    each_data_shape = (1, 7, *patch_size)
    each_data = np.zeros(each_data_shape, dtype=np.float32) 
    each_data[0, 0:1] = label_cap_after_resized
    each_data[0, 1:2] = label_cap_semantic_after_resized
    each_data[0, 2:3] = label_fen_after_resized
    each_data[0, 3:4] = label_fen_semantic_after_resized
    each_data[0, 4:5] = pred_cap_after_resized
    each_data[0, 5:6] = new_artery_after_resized
    each_data[0, 6:7] = old_artery_after_resized

    print('saving npy...', each_data.shape)
    np.save(each_sample_fusion_save_path, each_data.astype(np.float32))
    
    return each_sample_fusion_save_path


def save_gt_attr_after_resize(split_flag='train', normalization_flag='roi_cap'):
    lists = prepare_dataV2(split_flag, normalization_flag)
    lines_plus = copy.deepcopy(lists)
    # print(lines_plus[0])
    # [data, venous_data, roi, artery, ace, cap, fen, mvi, only_id]
    save_root_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/preprocessed_data/nnUNetData_plans_v2.1_stage1'
    each_add = [split_flag, save_root_path]
    lines_plus = [each_line_plus + each_add for each_line_plus in  lines_plus] 
    # lines_plus_try = [lines_plus[0]]

    p = Pool(8)
    return_data = p.starmap(each_sample_generate, lines_plus) # 其实并不需要return data
    p.close()
    p.join()
    
    # only_data = return_data[0]
    # each_sample_fusion_save_path, each_pkl_path = only_data[0], only_data[1]


def get_uid_to_id_dict():
    data_csv_root_path = '/GPFS/data/yuhangzhou/yuhangzhou/nnunet_dataset/nnUNet_raw/nnUNet_raw_data/Task300_liverMVI/log_loader.csv'
    uid_to_id = {}

    with open(data_csv_root_path, 'r') as f:
        reader = csv.reader(f)
        for i in reader:
            if i[1] == 'UID':
                continue
            uid_to_id[i[1]] = i[0]
    return uid_to_id

def get_uid_to_split_dict():
    data_csv_root_path = '/GPFS/data/yuhangzhou/yuhangzhou/nnunet_dataset/nnUNet_raw/nnUNet_raw_data/Task300_liverMVI/log_loader.csv'
    uid_to_split = {}

    with open(data_csv_root_path, 'r') as f:
        reader = csv.reader(f)
        for i in reader:
            if i[1] == 'UID':
                continue
            uid_to_split[i[1]] = i[-2]
    return uid_to_split

def get_id_to_split_dict():
    data_csv_root_path = '/GPFS/data/yuhangzhou/yuhangzhou/nnunet_dataset/nnUNet_raw/nnUNet_raw_data/Task300_liverMVI/log_loader.csv'
    id_to_split = {}

    with open(data_csv_root_path, 'r') as f:
        reader = csv.reader(f)
        for i in reader:
            if i[1] == 'UID':
                continue
            id_to_split[i[0]] = i[-2]
    return id_to_split


def slice_to_nii(save_pred_path, uid, type_flag):
    nii_root_path = '/GPFS/medical/private/jiangsu_liver/第三批-PHCC_converted'
    # ref_nii_path = join(nii_root_path, uid, uid+'_cap_5mm.nii') # label的spacing为1 ？！！！
    ref_nii_path = join(nii_root_path, uid, uid+'_img_5mm.nii')

    ref = sitk.ReadImage(ref_nii_path)
    npy_ref = sitk.GetArrayFromImage(ref)
    itk_spacing = ref.GetSpacing()
    itk_origin = ref.GetOrigin()
    itk_direction = ref.GetDirection()

    each_sample_slices = subfiles(join(save_pred_path, uid), join=True, suffix='.npz')
    npy_seg_new = np.zeros_like(npy_ref)
    # 得到有包膜的slice的序列
    each_sample_slice_th_list = [int(each_sample_slice.split('/')[-1][:-4].split('_')[-1]) for each_sample_slice in each_sample_slices]
    for each_sample_slice_th_th in range(len(each_sample_slice_th_list)):
        each_sample_slice_th = each_sample_slice_th_list[each_sample_slice_th_th]
        npy_seg_new[each_sample_slice_th] = np.load(each_sample_slices[each_sample_slice_th_th])[type_flag]

    return npy_seg_new, itk_spacing


def resample_venous_5mm_to_1mm_seg(uid):
    
    #ori_gt (医生版之前已使用)
    # our_gt, our_semantic, our_pred
    # each_sample_generate after resize

    # 先转nii
    load_5mm_root_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/cap_fen_5mm_slice'
    save_1mm_nii_root_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/cap_fen_1mm'
    maybe_mkdir_p(save_1mm_nii_root_path)

    type_flag = 'label_cap'
    

     # label_cap_semantic
    # our_pred 本身就有nii
    npy_seg, venous_spacing = slice_to_nii(load_5mm_root_path, uid, type_flag)


    uid_to_id = get_uid_to_id_dict()
    uid_to_split = get_uid_to_split_dict()
    print(uid_to_id[uid])

    raw_root_path = '/GPFS/data/yuhangzhou/yuhangzhou/nnunet_dataset/nnUNet_raw/nnUNet_raw_data/Task300_liverMVI'
    if uid_to_split[uid] == 'train':
        ref_artery = join(raw_root_path, 'imagesTr', uid_to_id[uid]+'_0000.nii.gz')
    elif uid_to_split[uid] == 'valid':
        ref_artery = join(raw_root_path, 'imagesTv', uid_to_id[uid]+'_0000.nii.gz')
    elif uid_to_split[uid] == 'test':
        ref_artery = join(raw_root_path, 'imagesTs', uid_to_id[uid]+'_0000.nii.gz')
    else:
        raise ValueError

    ref_artery_itk = sitk.ReadImage(ref_artery)
    if uid_to_id[uid] == '2285036':
        ref_artery_npy = sitk.GetArrayFromImage(ref_artery_itk)[:-1]
    else:
        ref_artery_npy = sitk.GetArrayFromImage(ref_artery_itk)[:-2]

    itk_spacing = ref_artery_itk.GetSpacing()
    itk_origin = ref_artery_itk.GetOrigin()
    itk_direction = ref_artery_itk.GetDirection()

    original_spacing = venous_spacing #npy_seg.GetSpacing()
    # new_shape = ref_artery_npy.shape
    target_spacing = ref_artery_itk.GetSpacing()


    if len(npy_seg.shape) == 3:
        npy_seg = npy_seg[np.newaxis, :]

    print('resample... ', uid)
    print('before size: ', npy_seg.shape)
    print('target_spacing: ', target_spacing)
    print('original_spacing: ', original_spacing)
    print('target_size: ', ref_artery_npy.shape) # new_shape

    max_spacing_axis = np.argmax(original_spacing)
    remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
    transpose_forward = [max_spacing_axis] + remaining_axes

    original_spacing = np.array(original_spacing)[transpose_forward]

    seg = npy_seg
    if seg is not None:
        assert len(seg.shape) == 4, "seg must be c x y z"
    shape = np.array(seg[0].shape)
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)  #spacing to target_spacing(spacing统一)后的size
    # new_shape = ref_artery_npy.shape

    if get_do_separate_z(original_spacing, 3):  #original_spacing是否有相差3倍以上的
        do_separate_z = True
        axis = get_lowres_axis(original_spacing)  #spacing最大的轴
    elif get_do_separate_z(target_spacing, 3): #target_spacing是否有相差3倍以上的
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
    each_venous_1mm_npy = resample_data_or_seg(seg, new_shape, True, axis, 1, do_separate_z, cval=-1, order_z=0)



    each_venous_1mm_npy = each_venous_1mm_npy.squeeze()
    print('after size: ', each_venous_1mm_npy.shape)
    # assert each_venous_1mm_npy.shape == ref_artery_npy.shape

    npy_seg = sitk.GetImageFromArray(each_venous_1mm_npy)
    npy_seg.SetSpacing(itk_spacing)
    npy_seg.SetOrigin(itk_origin)
    npy_seg.SetDirection(itk_direction)

    output_nii_path = join(save_1mm_nii_root_path, uid+'_'+type_flag+'_ori.nii.gz')
    print('saving... ', output_nii_path)
    sitk.WriteImage(npy_seg, output_nii_path)
    # break

#ori_gt (医生版之前已使用)
# our_gt, our_semantic, our_pred
# each_sample_generate after resize

def resample_venous_5mm_to_1mm_seg_plus(uid):
    # semantic 意味着向roi看齐了 ｜ 没有semantic 意味着 原始的cap label 经过腐蚀膨胀处理
    type_flag_list = ['label_cap', 'label_cap_semantic', 'pred_cap', 'label_fen', 'label_fen_semantic']
    save_1mm_nii_root_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/cap_fen_1mm'
    final_target_root_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/preprocessed_data/nnUNetData_plans_v2.1_stage1'
    maybe_mkdir_p(save_1mm_nii_root_path)

    uid_to_id = get_uid_to_id_dict()
    uid_to_split = get_uid_to_split_dict()
    print(uid, uid_to_id[uid])
    raw_root_path = '/GPFS/data/yuhangzhou/yuhangzhou/nnunet_dataset/nnUNet_raw/nnUNet_raw_data/Task300_liverMVI'

    for type_flag in type_flag_list:
        if type_flag == 'pred_cap':
            load_5mm_root_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/cap_fen_5mm_slice_pred_nii' 
            seg_itk = sitk.ReadImage(join(load_5mm_root_path, uid+'.nii.gz'))
            npy_seg = sitk.GetArrayFromImage(seg_itk)
            venous_spacing = seg_itk.GetSpacing()
        else:
            load_5mm_root_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/cap_fen_5mm_slice' # npz
            npy_seg, venous_spacing = slice_to_nii(load_5mm_root_path, uid, type_flag)

        original_spacing = venous_spacing

        if uid_to_split[uid] == 'train':
            ref_artery = join(raw_root_path, 'imagesTr', uid_to_id[uid]+'_0000.nii.gz')
            final_npy = np.load(join(final_target_root_path, 'imagesTr', '_roi', uid_to_id[uid]+'.npy'))[0]
        elif uid_to_split[uid] == 'valid':
            ref_artery = join(raw_root_path, 'imagesTv', uid_to_id[uid]+'_0000.nii.gz')
            final_npy = np.load(join(final_target_root_path, 'imagesTv', '_roi', uid_to_id[uid]+'.npy'))[0]
        elif uid_to_split[uid] == 'test':
            ref_artery = join(raw_root_path, 'imagesTs', uid_to_id[uid]+'_0000.nii.gz')
            final_npy = np.load(join(final_target_root_path, 'imagesTs', '_roi', uid_to_id[uid]+'.npy'))[0]
        else:
            raise ValueError
        # print('final_npy: ', final_npy.shape)

        ref_artery_itk = sitk.ReadImage(ref_artery)
        if uid_to_id[uid] == '2285036': # PH1164
            ref_artery_npy = sitk.GetArrayFromImage(ref_artery_itk)[:-1]
        else:
            ref_artery_npy = sitk.GetArrayFromImage(ref_artery_itk)[:-2]
        new_shape = ref_artery_npy.shape
        assert final_npy.shape == new_shape, print(uid, uid_to_id[uid], type_flag)

        itk_spacing = ref_artery_itk.GetSpacing()
        target_spacing = itk_spacing
        itk_origin = ref_artery_itk.GetOrigin()
        itk_direction = ref_artery_itk.GetDirection()

        # print('resample... ', uid)
        # print('before size: ', npy_seg.shape)
        # print('target_spacing: ', target_spacing)
        # print('original_spacing: ', original_spacing)
        # print('target_size: ', ref_artery_npy.shape)

        max_spacing_axis = np.argmax(original_spacing)
        remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
        transpose_forward = [max_spacing_axis] + remaining_axes
        original_spacing = np.array(original_spacing)[transpose_forward]

        if len(npy_seg.shape) == 3:
            npy_seg = npy_seg[np.newaxis, :]
        seg = npy_seg
        if seg is not None:
            assert len(seg.shape) == 4, "seg must be c x y z"

        # shape = np.array(seg[0].shape)
        # new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)

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
        each_venous_1mm_npy = resample_data_or_seg(seg, new_shape, True, axis, 1, do_separate_z, cval=-1, order_z=0)

        each_venous_1mm_npy = each_venous_1mm_npy.squeeze()
        # print('after size: ', each_venous_1mm_npy.shape)

        assert each_venous_1mm_npy.shape == new_shape, print(uid, uid_to_id[uid], type_flag)

        npy_seg_new = sitk.GetImageFromArray(each_venous_1mm_npy)
        npy_seg_new.SetSpacing(itk_spacing)
        npy_seg_new.SetOrigin(itk_origin)
        npy_seg_new.SetDirection(itk_direction)

        output_nii_path = join(save_1mm_nii_root_path, type_flag, uid_to_id[uid]+'.nii.gz')
        maybe_mkdir_p(join(save_1mm_nii_root_path, type_flag))
        # print('saving... ', output_nii_path)
        sitk.WriteImage(npy_seg_new, output_nii_path)
        # break


def save_pred_attr_after_resize():
    ready_root_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/cap_fen_1mm'
    new_type_flag = ['label_cap', 'label_cap_semantic', 'pred_cap', 'label_fen', 'label_fen_semantic']
    ready_sample_list = subfiles(join(ready_root_path, new_type_flag[0]), join=True, suffix='.nii.gz')
    id_list = [ready_sample.split('/')[-1].split('.')[0] for ready_sample in ready_sample_list]
    new_type_flag_plus = ['labels_artery_new', 'labels_artery_old']
    id_to_split = get_id_to_split_dict()
    labels_artery_new_root_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/preprocessed_data/nnUNetData_plans_v2.1_stage1/labels_artery_new'

    labels_artery_old_root_path = '/GPFS/data/yuhangzhou/yuhangzhou/nnunet_dataset/nnUNet_raw/nnUNet_raw_data/Task300_liverMVI/labels_artery'

    root_save_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/preprocessed_data/nnUNetData_plans_v2.1_stage1/resized_data_pred'
    maybe_mkdir_p(root_save_path)
    maybe_mkdir_p(join(root_save_path, 'train'))
    maybe_mkdir_p(join(root_save_path, 'valid'))
    maybe_mkdir_p(join(root_save_path, 'test'))
    id_list.sort()
    lines_plus = []
    for each_id in id_list:
        # if each_id != '2285036':
        #     continue
        label_cap_path = join(ready_root_path, 'label_cap', each_id+'.nii.gz')
        label_cap_semantic_path = join(ready_root_path, 'label_cap_semantic', each_id+'.nii.gz')
        pred_cap_path = join(ready_root_path, 'pred_cap', each_id+'.nii.gz')
        label_fen_path = join(ready_root_path, 'label_fen', each_id+'.nii.gz')
        label_fen_semantic_path = join(ready_root_path, 'label_fen_semantic', each_id+'.nii.gz')

        each_split = id_to_split[each_id]
        labels_artery_new_path = join(labels_artery_new_root_path, each_split, each_id+'.npy')
        labels_artery_old_path = join(labels_artery_old_root_path, each_split, each_id+'.nii.gz')
        save_npy_path = join(root_save_path, each_split, each_id+'.npy')
        save_pkl_path = join(root_save_path, each_split, each_id+'.pkl')
        each_line = [label_cap_path, label_cap_semantic_path, pred_cap_path, label_fen_path, label_fen_semantic_path, labels_artery_new_path, labels_artery_old_path, each_id, save_npy_path, save_pkl_path]
        lines_plus.append(each_line)

    print(len(lines_plus))
    # print([lines_plus[0]])
    p = Pool(8)
    return_data = p.starmap(each_sample_generate_pred, lines_plus) # 其实并不需要return data
    p.close()
    p.join()

    # th=2 有artery


def show_train_data_in_nii(case_id):

    case_id = str(case_id)

    id_to_split = get_id_to_split_dict()
    split = id_to_split[case_id]
    
    show3d_each_sample_in_nii(case_id, split_flag=split, normalization_flag='roi_cap')
    show3d_each_sample_in_nii_pred(case_id, split_flag=split, normalization_flag='roi_cap')



if __name__ == '__main__':
    # get_rotation_size() # [102 102 102]

    # get_xyz_roi_range()
    # 234 9 61.83
    # 186 10 64.77
    # 175 11 63.39


    # save_gt_attr_after_resize(split_flag='train', normalization_flag='roi_cap')
    # save_gt_attr_after_resize(split_flag='valid', normalization_flag='roi_cap')
    # save_gt_attr_after_resize(split_flag='test', normalization_flag='roi_cap')

    # show3d_each_sample_in_nii(split_flag='train', normalization_flag='roi_cap')


    # =========================
    
    # pred_nii_path = '/GPFS/medical/private/jiangsu_liver/Task300_liverMVI/cap_fen_5mm_slice_pred_nii'
    # pred_nii = subfiles(pred_nii_path, join=False, suffix='.nii.gz')
    # uid_list = [[each_uid.split('.')[0]] for each_uid in pred_nii]
    # uid_list.sort()

    # p = Pool(1)
    # return_data = p.starmap(resample_venous_5mm_to_1mm_seg_plus, [['PH1025']]) # 其实并不需要return data
    # p.close()
    # p.join()
    


    # semantic 中间是有缝隙的

    # =========================
    # save_pred_attr_after_resize()


    # show3d_each_sample_in_nii(split_flag='test', normalization_flag='roi_cap')
    # show3d_each_sample_in_nii_pred(split_flag='test', normalization_flag='roi_cap')


    # python /DATA5_DB8/data/yuhangzhou/LIVER_project/CODE/run_preprocess_v2.py


    # error_list = ['1120185','1302779', '2233987', '1634814', '1488666', '2315443', '1745000', '2072521', '2171267', '2317728', '1832313']
    # for case_id in error_list:
    #     show_train_data_in_nii(case_id)

    show_train_data_in_nii('1040639')


