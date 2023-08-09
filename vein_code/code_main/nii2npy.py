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

# 静脉期
vein_raw_dataroot_path = './tednet/vein_raw_data'
vein_target_path = './tednet/vein_raw_data'
vein_root_id_list = os.listdir(vein_raw_dataroot_path)


w_roi = []
for each_id in vein_root_id_list:

    img_data_path = os.path.join(vein_raw_dataroot_path, each_id, each_id+'_img_5mm.nii')
    roi_data_path = os.path.join(vein_raw_dataroot_path, each_id, each_id+'_roi_5mm.nii')

    if os.path.exists(img_data_path) and os.path.exists(roi_data_path):

        img_itk = sitk.ReadImage(img_data_path)
        img_npy = sitk.GetArrayFromImage(img_itk)
        img_save_path = os.path.join(vein_target_path, each_id, each_id+'_img.npy')
        np.save(img_save_path, img_npy)

        roi_itk = sitk.ReadImage(roi_data_path)
        roi_npy = sitk.GetArrayFromImage(roi_itk)
        roi_save_path = os.path.join(vein_target_path, each_id, each_id+'_roi.npy')
        np.save(roi_save_path, roi_npy)

        roi_mask = copy.deepcopy(roi_npy) > 0
        img_roi_npy = copy.deepcopy(img_npy)
        img_roi_voxels = list(img_roi_npy[roi_mask][::10])
        w_roi += img_roi_voxels

median_roi, mean_roi, sd_roi, mn_roi, mx_roi, percentile_99_5_roi, percentile_00_5_roi = _compute_stats(w_roi)

for each_id in vein_root_id_list:
    
    img_data_path = os.path.join(vein_raw_dataroot_path, each_id, each_id+'_img_5mm.nii')
    roi_data_path = os.path.join(vein_raw_dataroot_path, each_id, each_id+'_roi_5mm.nii')

    if os.path.exists(img_data_path) and os.path.exists(roi_data_path):
        img_itk = sitk.ReadImage(img_data_path)
        img_npy = sitk.GetArrayFromImage(img_itk)
        img_roi_npy = copy.deepcopy(img_npy)
        data_roi = np.clip(img_roi_npy, percentile_00_5_roi, percentile_99_5_roi)
        data_roi = (data_roi - mean_roi) / sd_roi

        img_roi_save_path = os.path.join(vein_target_path, each_id, each_id+'_img_roi.npy')
        np.save(img_roi_save_path, data_roi)

print(median_roi, mean_roi, sd_roi, mn_roi, mx_roi, percentile_99_5_roi, percentile_00_5_roi )
# 72.0 71.52487440950685 23.64727844379299 -883 974 130.0 8.0
print('nii2npy finish!')
    
    
    
    
