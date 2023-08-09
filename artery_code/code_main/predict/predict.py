from utils.file_ops import *
import numpy as np
from copy import deepcopy
from torch.cuda.amp import GradScaler, autocast 
from typing import Tuple, List, Union
import torch
from multiprocessing import Process, Queue
from collections import OrderedDict
from torch import nn
import SimpleITK as sitk
from time import time, sleep
from scipy.ndimage import label
import ast
import os

from models.unet_tmv import TMV_UNet
from models.unet_roi import ROI_UNet
from models.unet_ace import ACE_UNet

from postprocess.postprocesser import remove_all_but_the_largest_connected_component

class UNetPredictor(object):
    def __init__(self, Basic_Args):

        self.args = Basic_Args

        self.model_path = self.args.model_save_path
        self.input_folder = self.args.data_path
        self.output_niigz_folder = self.args.result_save_path

        self.step_size = self.args.step_size
        self.use_gaussian = self.args.use_gaussian
        self.do_tta = self.args.do_tta

        self.patch_size = np.array(self.args.patch_size).astype(int)
        self.amp_grad_scaler = GradScaler()

        if self.args.task_name == 'roi':
            self.network = ROI_UNet(num_classes=self.args.num_classes, act='leakyrelu', args=self.args)
        elif self.args.task_name == 'tmv':
            self.network = TMV_UNet(num_classes=self.args.num_classes, act='leakyrelu', args=self.args)
        elif self.args.task_name == 'ace':
            self.network = ACE_UNet(num_classes=self.args.num_classes, act='leakyrelu', args=self.args)
        elif self.args.task_name == 'vein_roi':
            self.network = ROI_UNet(num_classes=self.args.num_classes, act='leakyrelu', args=self.args)
        else:
            raise ValueError

        if torch.cuda.is_available():
            self.network.cuda()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def transform2npy(self):
        sample_list = os.listdir(self.input_folder)
        nii_sample_list = [os.path.join(self.input_folder, each_sample, 'artery_img.nii') for each_sample in sample_list]
        if self.args.task_name == 'vein_roi':
            nii_sample_list = [os.path.join(self.input_folder, each_sample, 'vein_img.nii') for each_sample in sample_list]
        for nii_sample in nii_sample_list:
            if os.path.exists(nii_sample):  
                nii_sample_itk = sitk.ReadImage(nii_sample)
                d = sitk.GetArrayFromImage(nii_sample_itk)[None]
                maybe_mkdir_p(os.path.join(self.output_niigz_folder, nii_sample.split('/')[-2]))
                save_path = os.path.join(self.output_niigz_folder, nii_sample.split('/')[-2], nii_sample.split('/')[-2]+'_artery_img_'+str(self.args.task_name)+'.npy')
                if self.args.task_name == 'vein_roi':
                    save_path = os.path.join(self.output_niigz_folder, nii_sample.split('/')[-2], nii_sample.split('/')[-2]+'_vein_img_'+str(self.args.task_name)+'.npy')

                if self.args.task_name == 'roi':
                    median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = 57.100666, 56.6966, 26.564444, -970.07886, 1009.7758, 121.83668487548829, -46.738822860717775
                elif self.args.task_name == 'ace':
                    median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = 74.73725, 74.626274, 22.728485, -878.7469, 938.94135, 130.14674110412597, -8.524003052711487
                elif self.args.task_name == 'tmv':
                    median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = 90.18779, 92.63411, 18.042349, 52.070435, 445.77646, 164.14702026367192, 62.45774742126465
                elif self.args.task_name == 'vein_roi':
                    median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = 72.0, 71.52487440950685, 23.64727844379299, -883, 974, 130.0, 8.0   
                else:
                    raise ValueError

                d = np.clip(d, percentile_00_5, percentile_99_5)
                d = (d - mean) / sd

                # if not os.path.exists(save_path):
                np.save(save_path, d)


    def run_predict(self):
        sample_list = os.listdir(self.input_folder)
        npy_sample_list = [os.path.join(self.output_niigz_folder, each_sample, each_sample+'_artery_img_'+str(self.args.task_name)+'.npy') for each_sample in sample_list]
        if self.args.task_name == 'vein_roi':
            npy_sample_list = [os.path.join(self.output_niigz_folder, each_sample, each_sample+'_vein_img_'+str(self.args.task_name)+'.npy') for each_sample in sample_list]

        checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.network.load_state_dict(checkpoint['net'])

        if 'amp_grad_scaler' in checkpoint.keys():
            self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        torch.cuda.empty_cache()

        for npy_sample in npy_sample_list:
            npy_sample_id = npy_sample.split('/')[-2]
            d = np.load(npy_sample)
            if 'vein' not in self.args.task_name:
                output_filename = os.path.join(self.output_niigz_folder, npy_sample_id, npy_sample_id + '_'+str(self.args.task_name)+'_artery.nii.gz')
            else:
                output_filename = os.path.join(self.output_niigz_folder, npy_sample_id, npy_sample_id + '_'+str(self.args.task_name)+'_vein.nii.gz')

            if self.args.ted:
                roi_path = os.path.join(self.output_niigz_folder, npy_sample_id, npy_sample_id + '_roi_artery.nii.gz')
                roi_itk = sitk.ReadImage(roi_path)
                roi_npy = sitk.GetArrayFromImage(roi_itk)[None]
                d = np.vstack((d, roi_npy))
                
            softmax_output = self.predict_preprocessed_data_return_seg_and_softmax(d, self.do_tta, (0,1,2), 
                        step_size=self.step_size, use_gaussian=self.use_gaussian)[1]

            self.save_segmentation_nifti_from_softmax(segmentation_softmax=softmax_output, out_fname=output_filename)


    def predict_preprocessed_data_return_seg_and_softmax(self, data, do_mirroring, mirror_axes, step_size,
                                                         use_gaussian):
        current_mode = self.network.training
        self.network.eval()
        ret = self.network.predict_3D(data, do_mirroring, mirror_axes, step_size, self.patch_size, use_gaussian, False)
        self.network.train(current_mode)
        return ret


    def save_segmentation_nifti_from_softmax(self, segmentation_softmax, out_fname):
        seg_old_size_postprocessed = segmentation_softmax.argmax(0)

        sample_list = os.listdir(self.input_folder)
        
        ref_filename  = join(self.input_folder, sample_list[0], 'artery_img.nii')
        if self.args.task_name == 'vein_roi':
            ref_filename  = join(self.input_folder, sample_list[0], 'vein_img.nii')

        raw_data = sitk.ReadImage(ref_filename)
        itk_spacing = raw_data.GetSpacing()
        itk_origin = raw_data.GetOrigin()
        itk_direction = raw_data.GetDirection()

        seg_old_size_postprocessed = self.run_postprocessing(seg_old_size_postprocessed, out_fname.split('/')[-2])

        seg_resized_itk = sitk.GetImageFromArray(seg_old_size_postprocessed.astype(np.uint8))
        seg_resized_itk.SetSpacing(itk_spacing)
        seg_resized_itk.SetOrigin(itk_origin)
        seg_resized_itk.SetDirection(itk_direction)
        
        sitk.WriteImage(seg_resized_itk, out_fname)


    def run_postprocessing(self, predict_npy, npy_sample_id=None):
        if 'roi' in self.args.task_name:
            predict_npy = remove_all_but_the_largest_connected_component(predict_npy)
        else:
            assert os.path.exists(os.path.join(self.output_niigz_folder, npy_sample_id, npy_sample_id + '_roi_artery.nii.gz'))
            roi_itk = sitk.ReadImage(os.path.join(self.output_niigz_folder, npy_sample_id, npy_sample_id + '_roi_artery.nii.gz'))
            roi_npy = sitk.GetArrayFromImage(roi_itk)

            if self.args.task_name == 'tmv':
                predict_npy = predict_npy * roi_npy
            elif self.args.task_name == 'ace':
                predict_npy = predict_npy * (1-roi_npy)
                predict_npy = remove_all_but_the_largest_connected_component(predict_npy)
            else:
                raise ValueError

        return predict_npy




if __name__=='__main__':
    pass