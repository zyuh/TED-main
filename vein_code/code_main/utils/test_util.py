# import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
import torch.nn as nn
import os

def getLargestCC(segmentation):
    segmentation_hcc = (segmentation==2).astype(int)

    labels = label(segmentation_hcc)
    if labels.max() == 0:
        return segmentation
    # assert(labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    segmentation[segmentation == 2] = 0
    segmentation[largestCC != 0] = 2
    return segmentation


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch) # modified
                    # ensemble
                    if num_classes>1:
                        y = nn.Softmax(dim=1)(y1)
                    else:
                        y = nn.Sigmoid()(y1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, 0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return np.array([dice, hd95, asd])
    else:
        return np.zeros(3)


def test_all_case_selected(net, base_dir, test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24, phase='V', skip_small=False, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    with open('./lists/' + '{}'.format(test_list), 'r') as f:
        image_list = f.readlines()

    total_metric = np.zeros((num_classes - 1, 3))
    print("Validation begin")
    valid_itr = 0
    normal_list = open(os.path.join(test_save_path, 'normal.txt'), 'w')
    FN_list = open(os.path.join(test_save_path, 'FN.txt'), 'w')
    Val_3070_list = open(os.path.join(test_save_path, 'Val_3070.txt'), 'w')
    for image_path in tqdm(image_list):
        data = np.load(base_dir + image_path.strip('\n') + '.npz')
        image = data['image' + phase]
        label = data['label' + phase]

        if skip_small:
            pad = [0, 0, 0]
            tempL = np.nonzero((label==2).astype(int))
            bbox = [[np.min(tempL[0]), np.max(tempL[0])], \
                     [np.min(tempL[1]), np.max(tempL[1])], \
                     [np.min(tempL[2]), np.max(tempL[2])]]
            W, H, D = bbox[0][1] - bbox[0][0], bbox[1][1] - bbox[1][0], bbox[2][1] - bbox[2][0]
            if W < 30 and H < 30 and D < 30:
                continue
            else:
                pass
            if W > 70 and H > 70 and D > 70:
                continue
            else:
                valid_itr += 1
                Val_3070_list.write(image_path.strip('\n'))
                Val_3070_list.write('\n')

        if np.sum(label==2) < 50:
            normal_list.write(image_path.strip('\n'))
            normal_list.write('\n')
            continue
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction) # have to modified

        if cal_metric(label == 2, prediction == 2)[0] == 0:
            FN_list.write(image_path.strip('\n'))
            FN_list.write('\n')

        for i in range(1, num_classes):
            total_metric[i - 1, :] += cal_metric(label == i, prediction == i)

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32),
                                     np.eye(4)), test_save_path + image_path.strip('\n') + '_pred.nii.gz')
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(
                4)), test_save_path + image_path.strip('\n') + '_img.nii.gz')
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(
                4)), test_save_path + image_path.strip('\n') + '_gt.nii.gz')

    print("testing end")
    print('valid iteration', valid_itr)
    return total_metric / valid_itr
