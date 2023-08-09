import math

import torch.nn as nn
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1, sdf_flag=False):
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
                    if sdf_flag:
                        _, y1, _ = net(test_patch)
                    else:
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
        asd = metric.binary.asd(pred, gt)
        return np.array([dice, asd])
    else:
        return np.zeros(2)



def test_all_case(net, base_dir, test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24, phase='V', onlinecrop_flag=False, non_oracle=False, skip_normal=False, sdf_flag=False):
    with open('./lists/' + '{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    if "MSD" in base_dir:
        image_list = [base_dir + "/data/{}.h5".format(
            item.replace('\n', '')) for item in image_list]
    if "NIH" in base_dir:
        image_list = [base_dir + "/data/{}.h5".format(
            item.replace('\n', '')) for item in image_list]
    elif "LA" in base_dir:
        image_list = [base_dir + "/{}/mri_norm2.h5".format(
            item.replace('\n', '')) for item in image_list]
    else:
        pass

    total_metric = np.zeros((num_classes-1, 3))
    print("Validation begin")
    for image_path in tqdm(image_list):
        data = np.load(base_dir + image_path.strip('\n') + '.npz')
        image = data['image'+phase]
        if non_oracle:
            label_liver = data['label'+phase+'_liver']
            label_hcc = data['label'+phase+'_hcc']
            label = np.zeros(np.shape(label_liver))
            label[label_liver==1] = 1
            label[label_hcc==1] = 2
        else:
            label = data['label'+phase]

        if skip_normal:
            if np.sum(label==2) == 0:
                continue
            else:
                pass
        # online crop 
        if onlinecrop_flag == True:
            pad = [16, 16, 16]
            patch_size = 128
            tempL = np.nonzero(label)
            bbox = [[max(0, np.min(tempL[0])-pad[0]), min(label.shape[0], np.max(label[0])+pad[0])], \
                    [max(0, np.min(tempL[1])-pad[1]), min(label.shape[1], np.max(label[1])+pad[1])], \
                    [max(0, np.min(tempL)-pad[2]), min(label.shape[2], np.max(label[2])+pad[2])]]
            for i in range(3):
                if bbox[i][1] - bbox[i][0] < 128:
                    if bbox[i][0] + 128 >= label.shape[i]:
                        bbox[i][0] = bbox[i][1] - 128
                    else:
                        bbox[i][1] = bbox[i][0] + 128
            image = image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] 
            label = label[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]].astype(np.uint8)

        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, sdf_flag=sdf_flag)
        for i in range(1, num_classes):
            total_metric[i-1, :] += cal_metric(label == i, prediction == i)
    print("Validation end")
    return total_metric / len(image_list)

