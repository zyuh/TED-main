import os
import shutil
import SimpleITK as sitk
import numpy as np
import copy

def _compute_stats(voxels):
    if len(voxels) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    median = np.median(voxels)
    mean = np.mean(voxels)
    sd = np.std(voxels)
    mn = np.min(voxels)
    mx = np.max(voxels)
    percentile_99_5 = np.percentile(voxels, 99.5)
    percentile_00_5 = np.percentile(voxels, 00.5)
    return median, mean, sd, mn, mx, percentile_99_5, percentile_00_5

# 动脉期
# tmv_root_path = './tednet/494_hcc_InternalArtery_labeled'
ace_root_path = './tednet/494_hcc_artery_1mm_ACE'
tmv_adjusted_path = './tednet/predict_adjusted' 

artery_target_path = './tednet/artery_train_data'
shutil.rmtree(artery_target_path)

# tmv_root_id_list = os.listdir(tmv_root_path)
ace_root_id_list = os.listdir(ace_root_path)
tmv_adjusted_list = os.listdir(tmv_adjusted_path)

print(len(ace_root_id_list), len(tmv_adjusted_list))

each_id_num = 0
img_num = 0
roi_num = 0
ace_num = 0
tmv_num = 0

w_roi = []
w_ace = []
w_tmv = []
for each_id in ace_root_id_list:
    tmv_data_path = None

    each_id_num = each_id_num + 1
    each_id_dir = os.path.join(artery_target_path, each_id)
    os.makedirs(each_id_dir)
    for each_tmv_adjusted in tmv_adjusted_list:
        if (each_id in each_tmv_adjusted) and (each_tmv_adjusted[-4:]=='.nii'):
            tmv_data_path = os.path.join(tmv_adjusted_path, each_tmv_adjusted, each_tmv_adjusted)
            break

    img_data_path = os.path.join(ace_root_path, each_id, 'hcc_surg_artery.nii')
    roi_data_path = os.path.join(ace_root_path, each_id, 'hcc_surg_artery_roi.nii')
    ace_data_path = os.path.join(ace_root_path, each_id, 'periT_ACE.nii')


    if os.path.exists(img_data_path):
        img_itk = sitk.ReadImage(img_data_path)
        img_npy = sitk.GetArrayFromImage(img_itk)
        img_save_path = os.path.join(artery_target_path, each_id, each_id+'_img.npy')
        np.save(img_save_path, img_npy)
        img_num = img_num + 1
    else:
        continue

    if os.path.exists(roi_data_path):
        roi_itk = sitk.ReadImage(roi_data_path)
        roi_npy = sitk.GetArrayFromImage(roi_itk)
        roi_save_path = os.path.join(artery_target_path, each_id, each_id+'_roi.npy')
        np.save(roi_save_path, roi_npy)
        roi_num = roi_num + 1

        roi_mask = copy.deepcopy(roi_npy) > 0
        img_roi_npy = copy.deepcopy(img_npy)
        img_roi_voxels = list(img_roi_npy[roi_mask][::10])
        w_roi += img_roi_voxels

    if os.path.exists(ace_data_path):
        ace_itk = sitk.ReadImage(ace_data_path)
        ace_npy = sitk.GetArrayFromImage(ace_itk)
        ace_save_path = os.path.join(artery_target_path, each_id, each_id+'_ace.npy')
        np.save(ace_save_path, ace_npy)
        ace_num = ace_num + 1
    
        ace_mask = copy.deepcopy(ace_npy) > 0
        img_ace_npy = copy.deepcopy(img_npy)
        img_ace_voxels = list(img_ace_npy[ace_mask][::10])
        w_ace += img_ace_voxels

    if tmv_data_path is not None:
        tmv_itk = sitk.ReadImage(tmv_data_path)
        tmv_npy = sitk.GetArrayFromImage(tmv_itk)
        tmv_save_path = os.path.join(artery_target_path, each_id, each_id+'_tmv.npy')
        np.save(tmv_save_path, tmv_npy)
        tmv_num = tmv_num + 1

        tmv_mask = copy.deepcopy(tmv_npy) > 0
        img_tmv_npy = copy.deepcopy(img_npy)
        img_tmv_voxels = list(img_tmv_npy[tmv_mask][::10])
        w_tmv += img_tmv_voxels

median_roi, mean_roi, sd_roi, mn_roi, mx_roi, percentile_99_5_roi, percentile_00_5_roi = _compute_stats(w_roi)
median_ace, mean_ace, sd_ace, mn_ace, mx_ace, percentile_99_5_ace, percentile_00_5_ace = _compute_stats(w_ace)
median_tmv, mean_tmv, sd_tmv, mn_tmv, mx_tmv, percentile_99_5_tmv, percentile_00_5_tmv = _compute_stats(w_tmv)

# print(median_roi, mean_roi, sd_roi, mn_roi, mx_roi, percentile_99_5_roi, percentile_00_5_roi)
# 57.100666 56.6966 26.564444 -970.07886 1009.7758 121.83668487548829 -46.738822860717775
# print(median_ace, mean_ace, sd_ace, mn_ace, mx_ace, percentile_99_5_ace, percentile_00_5_ace)
# 74.73725 74.626274 22.728485 -878.7469 938.94135 130.14674110412597 -8.524003052711487
# print(median_tmv, mean_tmv, sd_tmv, mn_tmv, mx_tmv, percentile_99_5_tmv, percentile_00_5_tmv)
# 90.18779 92.63411 18.042349 52.070435 445.77646 164.14702026367192 62.45774742126465


for each_id in ace_root_id_list:
    img_data_path = os.path.join(ace_root_path, each_id, 'hcc_surg_artery.nii')

    if os.path.exists(img_data_path):
        img_itk = sitk.ReadImage(img_data_path)
        img_npy = sitk.GetArrayFromImage(img_itk)
        # img_save_path = os.path.join(artery_target_path, each_id, each_id+'_img.npy')
        # np.save(img_save_path, img_npy)
        # img_num = img_num + 1

        img_roi_npy = copy.deepcopy(img_npy)
        img_ace_npy = copy.deepcopy(img_npy)
        img_tmv_npy = copy.deepcopy(img_npy)

        data_roi = np.clip(img_roi_npy, percentile_00_5_roi, percentile_99_5_roi)
        data_roi = (data_roi - mean_roi) / sd_roi
        data_ace = np.clip(img_ace_npy, percentile_00_5_ace, percentile_99_5_ace)
        data_ace = (data_ace - mean_ace) / sd_ace
        data_tmv = np.clip(img_tmv_npy, percentile_00_5_tmv, percentile_99_5_tmv)
        data_tmv = (data_tmv - mean_tmv) / sd_tmv

        img_roi_save_path = os.path.join(artery_target_path, each_id, each_id+'_img_roi.npy')
        np.save(img_roi_save_path, data_roi)
        img_ace_save_path = os.path.join(artery_target_path, each_id, each_id+'_img_ace.npy')
        np.save(img_ace_save_path, data_ace)
        img_tmv_save_path = os.path.join(artery_target_path, each_id, each_id+'_img_tmv.npy')
        np.save(img_tmv_save_path, data_tmv)
    else:
        continue


print(each_id_num, img_num, roi_num, ace_num, tmv_num)
    
    
    
    
