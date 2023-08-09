import os
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn as nn
import cv2
from skimage import measure
import itertools
import matplotlib.pyplot as plt
import SimpleITK as sitk 
import copy

import segmentation_models.segmentation_models_pytorch as smp
from dataloaders import Cap_2d_full2021_fpn_test
from utils.util import _recover_from_flatten


import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='./tednet/train_data', help='Name of Experiment')
parser.add_argument('--load_snapshot_path', type=str,
                    default='./vein_code/model/cap_try_dcap_lr0.01_bs64_epo300_ncls4_joint/epoch_299.pth', help='load snapshot for fine tuninng or testing!')

parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu')
parser.add_argument('--ring_width', type=int, default=150, help='max ring width') # default 0, but not meet 5x5 blur
parser.add_argument('--n_ray', type=int, default=90, help='number of rays')
parser.add_argument('--n_classes', type=int, default=4, help='random seed')
parser.add_argument('--data_type', type=str, default='cap', help='cap, fen')

parser.add_argument('--is_tumor_seg', action='store_true', help='jointly train tumor seg and seq')
parser.add_argument('--is_hcccap', action='store_true', help='jointly train hcc and cap') 
parser.add_argument('--is_testvertex', action='store_true', help='using predicted vertex to sampling feature in test phase')
parser.add_argument('--is_recover_rect', action='store_true', help='recover_rect')

parser.add_argument('--test_middle_save', type=str, default='', help='Name of Experiment')
parser.add_argument('--test_save', type=str, default='', help='Name of Experiment')

args = parser.parse_args()
batch_size = args.batch_size

def vis(array, name_b, snapshot_path, vis_center=False, center=None, center_gt=None):
    array_vis = np.zeros((array.shape[0], array.shape[1], 3))
    # opencv: bgr
    array_vis[array == 1, 1] = 255 # green
    array_vis[array == 2, 0] = 255 # blue
    array_vis[array == 3, 2] = 255 # red
    if vis_center:
        cX, cY = center
        array_vis[cX:cX + 5, cY:cY + 5, 1] = 255
        if center_gt is not None:
            cX_gt, cY_gt = center
            array_vis[cX_gt:cX_gt + 5, cY_gt:cY_gt + 5, 2] = 255  # red
    if not os.path.exists(os.path.join(snapshot_path, name_b.split('_')[0], args.data_type)):
        os.makedirs(os.path.join(snapshot_path, name_b.split('_')[0], args.data_type))
    cv2.imwrite(os.path.join(snapshot_path, name_b.split('_')[0], args.data_type, name_b +'_'+args.data_type +'.png'), array_vis)


def gen_seqlabel_cartcoord_torch(cap_degree, args=None):
    B = len(cap_degree)
    seq_label = torch.zeros((B, 90,))
    vertexs = torch.zeros((B, 90, 2))
    for b in range(B):
        for d in range(90):
            roi_nonzero = torch.nonzero((cap_degree[b]==d+1).float(), as_tuple=True) # (2, N)
            try:
                idx = np.random.randint(0, len(roi_nonzero[0]))
            except:
                print("tumor label not found!")
                cv2.imwrite(args.vis_dir + "/" + args.name[b] + "______degree_slice.png", (cap_degree[b].cpu().detach().numpy() * (255 / 90)).astype(np.uint8))
                try:
                    roi_nonzero = torch.nonzero((cap_degree[b-1] == d+1).float(), as_tuple=True)
                    idx = np.random.randint(0, len(roi_nonzero[0]))
                except:
                    try:
                        print("cap label not found in this slice, staring search next cap slice")
                        roi_nonzero = torch.nonzero((cap_degree[b+1] == d+1).float(), as_tuple=True)
                        idx = np.random.randint(0, len(roi_nonzero[0]))
                    except:
                        try:
                            roi_nonzero = torch.nonzero((cap_degree[b-2] == d+1).float(), as_tuple=True)
                            idx = np.random.randint(0, len(roi_nonzero[0]))
                        except:
                            roi_nonzero = torch.nonzero((cap_degree[b+2] == d+1).float(), as_tuple=True)
                            idx = np.random.randint(0, len(roi_nonzero[0]))
                            
            vertexs[b, d, 0], vertexs[b, d, 1] = roi_nonzero[0][idx], roi_nonzero[1][idx]
    vertexs = (vertexs / 4.0).int()
    if 90 % args.n_ray != 0:
        assert 1==2, "90%args.n_ray!=0"

    return vertexs


def cart2polar(mask, M, cap, recover_flag=False, cap2=None, prgt_flag=False, args=None):
    if cap2 is not None:
        prgt_flag = True
        flatten_prgt = np.zeros((360, args.ring_width))

    cart_x, cart_y = np.nonzero(mask)
    cart_x, cart_y = np.array(cart_x).astype(np.float32), np.array(cart_y).astype(np.float32)
    cY, cX = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    cart_offset_x, cart_offset_y = cart_x - cX, cart_y - cY
    r1, theta1 = cv2.cartToPolar(cart_offset_x, cart_offset_y, angleInDegrees=True)

    # print(np.array(r1).shape, np.array(theta1).shape)
    r1, theta1 = np.array(r1)[:, 0], np.array(theta1)[:, 0]
    theta1_int = [np.round(x) for x in theta1]
    count_len = []
    flatten = np.zeros((360, args.ring_width))
    # print('range theta', np.int(min(theta1_int)), 1 + np.int(max(theta1_int)))
    for idx in np.arange(0, 360):  # np.arange(np.int(min(theta1_int)), 1 + np.int(max(theta1_int))):
        tmp_r = np.array([r1[i] for i in np.where(theta1_int == np.float32(idx))])
        tmp_t = np.array([theta1[i] for i in np.where(theta1_int == np.float32(idx))])
        sortind = np.argsort(tmp_r)[0]
        tmp_r = np.array([tmp_r[0][id] for id in sortind])
        tmp_t = np.array([tmp_t[0][id] for id in sortind])  # the same number
        if len(tmp_r) == 0: continue  # filter !!!!
        count_len.append(len(tmp_r))
        x_new, y_new = cv2.polarToCart(tmp_r, tmp_t, angleInDegrees=True)
        x_cart_new, y_cart_new = x_new + cX, y_new + cY
        x_cart_new = [np.uint(np.round(x)) for x in x_cart_new]
        y_cart_new = [np.uint(np.round(y)) for y in y_cart_new]
        # generate a prediction tensor!
        tmp_list = np.zeros(args.n_classes - 1)
        for i in range(len(y_cart_new)):
            flatten[idx][i] = cap[np.int(x_cart_new[i])][np.int(y_cart_new[i])]
            if prgt_flag:
                flatten_prgt[idx][i] = cap2[np.int(x_cart_new[i])][np.int(y_cart_new[i])]

    if prgt_flag and recover_flag:  # pr + prgt
        return flatten, flatten_prgt, r1, theta1
    elif not prgt_flag and recover_flag:
        return flatten, r1, theta1
    elif not prgt_flag and not recover_flag:  # gt
        return flatten
    else:
        assert 1 == 2, 'error return'




if __name__ == "__main__":
    '''
    raw_samples_list = os.listdir(args.dataset)
    for each_sample in raw_samples_list:
        raw_nii_path = os.path.join(args.dataset, each_sample, 'vein_img.nii')
        roi_nii_path = os.path.join(args.test_middle_save, each_sample, each_sample+'_roi.nii')
        intermediate_npz_root_path = os.path.join(args.test_middle_save, each_sample)
        if not os.path.exists(intermediate_npz_root_path):
            os.makedirs(intermediate_npz_root_path)
        raw_img = sitk.GetArrayFromImage(sitk.ReadImage(raw_nii_path))
        raw_roi = sitk.GetArrayFromImage(sitk.ReadImage(roi_nii_path))
        # print(raw_img.shape)
        for each_slice in raw_img:
            if np.sum(raw_roi[each_slice])>100:
                img_slice = raw_img[each_slice]
                roi_slice = raw_roi[each_slice]
                np.savez(os.path.join(intermediate_npz_root_path, each_sample +'_slice_'+ each_slice + '.npz'), image=img_slice, roi=roi_slice)
    '''

    net = smp.PHCC_FPN('resnet50', in_channels=1, classes=4, encoder_weights='imagenet', 
            attention=None, is_tumor_seg=args.is_tumor_seg, is_testvertex=args.is_testvertex)
    net = net.cuda()
    state_dict = torch.load(args.load_snapshot_path)
    net.load_state_dict(state_dict)
    net.eval()


    # build data loader
    Cap_2d = Cap_2d_full2021_fpn_test
    cfg_test = Cap_2d.Config(datapath=args.dataset, mode='test', data_type=args.data_type, 
        batch=args.batch_size, n_classes=4, is_recover_rect=args.is_recover_rect, 
        list_path=args.test_middle_save, is_boundary=False, is_hcccap=args.is_hcccap, is_ctc=False, 
        is_transformer=False, is_multidoc=False, doc_id=1000, is_transformer_seg=False)

    db_test = Cap_2d.Data(cfg_test)
    testloader = DataLoader(db_test, collate_fn=db_test.testcollate, batch_size=cfg_test.batch, 
                shuffle=False, num_workers=4)


    for i_batch, sampled_batch in enumerate(testloader):
        inputs, label_hcc, cap_degree, rect, name = sampled_batch
        inputs = inputs.cuda().float()

        if args.is_testvertex:
            vertexs = torch.zeros((batch_size, 90, 2))
            out_seq, out_tumor = net(inputs, vertexs)
        else:
            vertexs = gen_seqlabel_cartcoord_torch(cap_degree, args)
            out_seq, out_tumor = net(inputs, vertexs)
        seq_pred = torch.argmax(torch.softmax(out_seq, dim=1), dim=1)

        mask_tumor = torch.argmax(torch.softmax(out_tumor, dim=1), dim=1).cpu().detach().numpy()
        label_hcc = label_hcc.numpy()

        flatten_seq = seq_pred.unsqueeze(1).unsqueeze(3).repeat(1,1,1, args.ring_width)
        flatten_seq = nn.Upsample(size=(360, args.ring_width), mode='nearest')(flatten_seq.float()).cpu().detach().numpy()

        for b in range(len(out_seq)):
            if np.sum(mask_tumor[b]>0) != 0:
                M = cv2.moments((mask_tumor[b]>0).astype(np.float))
            else:
                M = cv2.moments((label_hcc[b]>0).astype(np.float))

            cY, cX = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            dilation = cv2.dilate(mask_tumor[b].astype(np.uint8), kernel=np.ones((3, 3), np.uint8), iterations=1)
            erosion = cv2.erode(mask_tumor[b].astype(np.uint8), kernel=np.ones((3, 3), np.uint8), iterations=1)
            hcc_cvt_edge_b = (cv2.GaussianBlur((dilation - erosion).astype(np.float), (5, 5), 0) > 0).astype(np.float)
            # print(name[b], np.sum(hcc_cvt_edge_b), M)
            if np.sum(hcc_cvt_edge_b) == 0:
                continue

            _, r1, theta1 = cart2polar(hcc_cvt_edge_b, M, hcc_cvt_edge_b, recover_flag=True, args=args)  # take 0.6 s
            pred_recover_tumor_b = _recover_from_flatten(flatten_seq[b][0]+1, np.zeros_like(mask_tumor[b]), M, r1, theta1)

            p0, p1, p2, p3 = rect[b][0]
            p0 = np.array(p0)
            p1 = np.array(p1)
            p2 = np.array(p2)
            p3 = np.array(p3)
            pred_recover_tumor_b_copy = copy.deepcopy(pred_recover_tumor_b)
            pred_recover_tmor_ori_b = np.zeros((512, 512))
            pred_recover_tumor_b = cv2.resize(pred_recover_tumor_b, dsize=(p1-p0, p3-p2), interpolation=cv2.INTER_NEAREST)
            pred_recover_tmor_ori_b[p0:p1, p2:p3] = pred_recover_tumor_b
 
            vis(pred_recover_tmor_ori_b, name[b], args.test_middle_save)
            # print(os.path.join(args.test_middle_save, name[b].split('_')[0], name[b]+'_'+args.data_type+'.npy'))
            np.save(os.path.join(args.test_middle_save, name[b].split('_')[0], name[b]+'_'+args.data_type+'.npy'), pred_recover_tmor_ori_b)

            image = inputs[b][0].cpu().detach().numpy()
            roi = label_hcc[b] * 255.0
            image = (image - image.min()) / (image.max() - image.min()) * 255.0
            cv2.imwrite(os.path.join(args.test_middle_save, name[b].split('_')[0], args.data_type, name[b] + '_img.png'), image.astype(np.uint8))
            cv2.imwrite(os.path.join(args.test_middle_save, name[b].split('_')[0], args.data_type, name[b] + '_roi.png'), roi.astype(np.uint8))
            image_vis = np.zeros((image.shape[0], image.shape[1], 3))
            image_vis[:,:,0] = image
            image_vis[:,:,1] = image
            image_vis[:,:,2] = image

            image_vis[pred_recover_tumor_b_copy == 1,1] = 255
            image_vis[pred_recover_tumor_b_copy == 1,0] = 0
            image_vis[pred_recover_tumor_b_copy == 1,2] = 0
            image_vis[pred_recover_tumor_b_copy == 2,0] = 255
            image_vis[pred_recover_tumor_b_copy == 2,1] = 0
            image_vis[pred_recover_tumor_b_copy == 2,2] = 0
            image_vis[pred_recover_tumor_b_copy == 3,2] = 255
            image_vis[pred_recover_tumor_b_copy == 3,0] = 0
            image_vis[pred_recover_tumor_b_copy == 3,1] = 0

            cv2.imwrite(os.path.join(args.test_middle_save, name[b].split('_')[0], args.data_type, name[b] +'_'+args.data_type+ '_pred_img.png'), image_vis.astype(np.uint8))

if args.data_type == 'cap':
    print('predict cap finish!')
elif args.data_type == 'fen':
    print('predict fen finish!')
