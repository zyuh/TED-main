import csv
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.file_ops import *
import argparse

from predict.predict import UNetPredictor
from predict.evaluation import runningScore

def test(Basic_Args):
    maybe_mkdir_p(Basic_Args.result_save_path)

    if Basic_Args.gt_label_path is not None:
        LOG_CSV = os.path.join(Basic_Args.result_save_path, 'roi_test_dice.csv')
        LOG_CSV_HEADER = ['case_name', 'dice_per_class']

        if os.path.exists(LOG_CSV):
            os.remove(LOG_CSV)

        with open(LOG_CSV, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(LOG_CSV_HEADER)

    predictor = UNetPredictor(Basic_Args)

    predictor.transform2npy()
    predictor.run_predict()
    
    if Basic_Args.gt_label_path is not None:
        ##=============================== compute dice ==========================##
        print('==> Start compute dice')
        running_eval = runningScore(Basic_Args)

        all_cls_dice = []
        all_cls_dice_post = []

        sample_list = os.listdir(output_nii_folder_after_postprocessing)
        label_post_preds_case = [join(output_nii_folder_after_postprocessing, each_sample, each_sample+'_tmv.nii.gz') for each_sample in sample_list if os.path.exists(join(output_nii_folder_after_postprocessing, each_sample, each_sample+'_tmv.nii.gz'))]

        sample_list = os.listdir(Basic_Args.gt_label_path)
        label_truth_case = [join(Basic_Args.gt_label_path, each_sample, 'hcc_surg_artery_B.nii') for each_sample in sample_list if os.path.exists(join(Basic_Args.gt_label_path, each_sample, 'hcc_surg_artery_roi.nii'))]
        
        confusion_matrix_post = np.zeros((len(label_post_preds_case), Basic_Args.num_classes, Basic_Args.num_classes))

        num = 0
        for ltc, lppc in zip(label_truth_case, label_post_preds_case):
            case_name  = ltc.split('/')[-2]
            running_eval.update(ltc, lppc)
            _, cls_dice_post = running_eval.get_scores()
            running_eval.reset()
            cls_dice_post, c_matrix_post = cls_dice_post['Class Dice'].tolist(), cls_dice_post['confusion_matrix']
            confusion_matrix_post[num] = c_matrix_post
            all_cls_dice_post.append(cls_dice_post)
            dc_msg = '         Dice per class post:   '.format(case_name)
            for j in range(len(cls_dice_post)):
                dc_msg += '{:.5f}, '.format(cls_dice_post[j])
            dc_msg = dc_msg[:-2]
            print(dc_msg)

            log_vector = [case_name] + cls_dice_post

            with open(LOG_CSV, 'a') as f:
                logwriter = csv.writer(f, delimiter=',')
                logwriter.writerow(log_vector)
            num = num + 1

        np.save(join(Basic_Args.result_save_path, 'confusion_matrix_post.npy'), confusion_matrix_post.astype(np.uint64))

        print('done...')
        # all_cls_dice_post = np.array(all_cls_dice_post)
        # all_cls_dice_post[all_cls_dice_post==0] = np.nan
        # all_cls_dice_post = np.nanmean(all_cls_dice_post, axis=0)
        all_cls_dice_post = np.mean(all_cls_dice_post, axis=0)

        print(all_cls_dice_post)
        msg = 'Mean Dice per class post:  '
        for i in range(len(all_cls_dice_post)):
            msg += '{:.5f}, '.format(all_cls_dice_post[i])
        msg = msg[:-2]
        print(msg)

        log_vector = ['mean'] + all_cls_dice_post.tolist()
        with open(LOG_CSV, 'a') as f:
            logwriter = csv.writer(f, delimiter=',')
            logwriter.writerow(log_vector)   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model_save_path', type=str, default=None)
    parser.add_argument('--result_save_path', type=str, default=None)
    parser.add_argument('--gt_label_path', type=str, default=None)
    parser.add_argument('--ted', action='store_true')
    parser.add_argument('--task_name', type=str, default='tmv')
    
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--patch_size', default=[64, 64, 64])

    parser.add_argument('--step_size', type=float, default=0.5)
    parser.add_argument('--use_gaussian', action='store_true')
    parser.add_argument('--do_tta', action='store_true')

    args = parser.parse_args()
    test(args)
    print('artery tmv predict finish!')
