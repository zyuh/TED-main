#!/usr/bin/python3
# coding=utf-8
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from operator import itemgetter


########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, body, edge):
        image = (image - self.mean) / self.std

        return image, mask / 255, body / 255, edge / 255


class MaskRandomCrop(object):
    def __call__(self, rect, image, mask, hcc_mask=None, cap_degree=None, cap_semantic=None, mode='train', cfg=None):
        H0, W0 = image.shape
        if mode == 'train':
            maxoffset = 60 # d 60
            minoffset = 10 # d 20
            offsetl = np.random.randint(minoffset, maxoffset)
            offsetr = np.random.randint(minoffset, maxoffset)
            offsett = np.random.randint(minoffset, maxoffset)
            offsetb = np.random.randint(minoffset, maxoffset)
            if cfg.is_transformer:
                offsetl, offsetl, offsett, offsetb = 20, 20, 20, 20

            p2 = max(rect[2] - offsetl, 0)
            p3 = min(rect[3] + offsetr, W0)
            p0 = max(rect[0] - offsett, 0)
            p1 = min(rect[1] + offsetb, H0)
        else:
            fix_offset = 20
            p2 = max(rect[2] - fix_offset, 0)
            p3 = min(rect[3] + fix_offset, W0)
            p0 = max(rect[0] - fix_offset, 0)
            p1 = min(rect[1] + fix_offset, H0)

        rect_w = p3 - p2
        rect_h = p1 - p0
        pad_l = abs(rect_w - rect_h)
        if rect_h > rect_w:
            p2 = max(p2 - int(pad_l / 2.0), 0)
            p3 = min(p2 + pad_l + rect_w, image.shape[1])
        else:
            p0 = max(p0 - int(pad_l / 2.0), 0)
            p1 = min(p0 + pad_l + rect_h, image.shape[0])
        # see Youbao's code if needing visualization!

        if cap_degree is not None and cap_semantic is not None:
            if cfg.is_recover_rect:
                return image[p0:p1, p2:p3], mask[p0:p1, p2:p3], hcc_mask[p0:p1, p2:p3], cap_degree[p0:p1,p2:p3], cap_semantic[p0:p1,p2:p3], [p0,p1,p2,p3]
            return image[p0:p1, p2:p3], mask[p0:p1, p2:p3], hcc_mask[p0:p1, p2:p3], cap_degree[p0:p1, p2:p3], cap_semantic[p0:p1, p2:p3]
        if hcc_mask is not None:
            return image[p0:p1, p2:p3], mask[p0:p1, p2:p3], hcc_mask[p0:p1, p2:p3]
        return image[p0:p1, p2:p3], mask[p0:p1, p2:p3], None  # image channel=1




class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, body, edge):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        body = cv2.resize(body, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        edge = cv2.resize(edge, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, body, edge


class ToTensor(object):
    def __call__(self, image, mask, body, edge):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        body = torch.from_numpy(body).unsqueeze(0)
        edge = torch.from_numpy(edge).unsqueeze(0)
        return image, mask, body, edge


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean = np.array([[[-61.3, -61.3, -61.3]]]).astype(np.float32)
        self.std = np.array([[[79.8, 79.8, 79.8]]]).astype(np.float32)
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None

def random_rot_flip(image, label, hcc_mask, cap_degree, cap_semantic):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()

    hcc_mask = np.rot90(hcc_mask, k)
    hcc_mask = np.flip(hcc_mask, axis=axis).copy()

    cap_degree = np.rot90(cap_degree, k)
    cap_degree = np.flip(cap_degree, axis=axis).copy()

    cap_semantic = np.rot90(cap_semantic, k)
    cap_semantic = np.flip(cap_semantic, axis=axis).copy()
    return image, label, hcc_mask, cap_degree, cap_semantic


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.normalize = Normalize(mean=cfg.mean, std=cfg.std)
        self.maskrandomcrop = MaskRandomCrop()

        self.resize = Resize(512, 512)
        self.samples = []
        self.pixel_mean, self.pixel_std = 42.63, 69.51  # loop all doctors all cap under clip(-100, 240)

        # cfg.mode = 'train' | 'val' | 'test'
        if self.cfg.mode == 'train':
            list_txt = os.path.join(self.cfg.list_path, self.cfg.data_type[-3:]+'_split_train_slices.txt')
        else:
            list_txt = os.path.join(self.cfg.list_path, self.cfg.data_type[-3:]+'_split_test_slices.txt')

        with open(list_txt, 'r') as lines:
            for line in lines:
                if line.strip('\r\n') == '':
                    continue
                self.samples.append(line.strip().strip('\r\n'))
        # print(len(self.samples), self.samples)

    def bbox2(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    def gen_seqlabel_cartcoord(cap_degree, cap_semantic):
        # can write to torch version
        seq_label = np.zeros((90,))
        vertexs = np.zeros((90,2))
        for d in range(90):
            # roi_degree = (cap_degree==i).astype(np.float)
            roi_nonzero = np.nonzero(cap_degree==d) # (2, N)
            idx = np.random.randint(0,len(roi_nonzero[0]))
            d_semantic = cap_semantic[roi_nonzero[0][idx], roi_nonzero[1][idx]]
            seq_label[d] = d_semantic
            vertexs[d, :] = [roi_nonzero[0][idx], roi_nonzero[1][idx]]

    def __getitem__(self, idx):
        name = self.samples[idx]
        each_uid = name[0:6]
        datapath = self.cfg.datapath
        dname = self.cfg.data_type[-3:]

        data_path = self.cfg.datapath + '/' + each_uid + '/' + name + '.npz'
        data = np.load(data_path)

        image, label = data['image'], data['label_'+dname]

        gt_c = np.ones(1, dtype=np.float32)[0]
        hcc_mask = data['label_roi'] if self.cfg.is_hcccap else None

        data_degree, data_semantic = data['label_'+dname+'_degree'], data['label_'+dname+'_semantic']
        rect = self.bbox2((label>0).astype(np.float)+(hcc_mask>0).astype(np.float))

        # gt_c = rect  # np.shape(image)

        image = np.clip(image, -100, 240)
        # image = (image - image.min()) / (image.max() - image.min())  # normalized
        # image = (image - np.mean(image)) / np.std(image)
        image = (image - self.pixel_mean) / self.pixel_std

        if self.cfg.is_recover_rect:
            image, label, hcc_mask, data_degree, data_semantic, rect2 = self.maskrandomcrop(rect, image, label, hcc_mask, data_degree, data_semantic, self.cfg.mode, self.cfg)
            gt_c = rect2
        else:
            image, label, hcc_mask, data_degree, data_semantic = self.maskrandomcrop(rect, image, label, hcc_mask, data_degree, data_semantic, self.cfg.mode, self.cfg)
        if self.cfg.mode=='train' and not self.cfg.is_recover_rect:
            image, label, hcc_mask, data_degree, data_semantic = random_rot_flip(image, label, hcc_mask, data_degree, data_semantic)
        return image, label, hcc_mask, data_degree, data_semantic, gt_c, name

    def collate(self, batch):
        # size = [224, 256, 288, 320, 352, 384, 416, 448, 480][np.random.randint(0, 9)]
        # Got it, size consistency in a batch
        size = [192, 224, 256, 288, 320][np.random.randint(0, 5)]
        if self.cfg.is_transformer:
            size = 96
        if self.cfg.is_transformer_seg:
            size = 224
        # size = 224
        # image, mask, gt_c, name = [list(item) for item in zip(*batch)]
        image, mask, hcc_mask, cap_degree, cap_semantic, gt_c, name = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i], dsize=(size, size), interpolation=cv2.INTER_NEAREST)  # nearest or one-hot!
            cap_degree[i] = cv2.resize(cap_degree[i], dsize=(size, size), interpolation=cv2.INTER_NEAREST)
            cap_semantic[i] = cv2.resize(cap_semantic[i], dsize=(size, size), interpolation=cv2.INTER_NEAREST)
            hcc_mask[i] = cv2.resize(hcc_mask[i], dsize=(size, size), interpolation=cv2.INTER_NEAREST)
        # image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)


        image = torch.from_numpy(np.stack(image, axis=0)).unsqueeze(1) # to tensor
        mask = torch.from_numpy(np.stack(mask, axis=0))# .unsqueeze(1)
        gt_c = torch.from_numpy(np.stack(gt_c, axis=0)).unsqueeze(1)
        hcc_mask = torch.from_numpy(np.stack(hcc_mask, axis=0))
        # print("hcc_mask", hcc_mask.shape)
        # print("cap mask", mask.shape)

        cap_degree = torch.from_numpy(np.stack(cap_degree, axis=0))
        cap_semantic = torch.from_numpy(np.stack(cap_semantic, axis=0))
        return image, mask, hcc_mask, cap_degree, cap_semantic, gt_c, name

    def testcollate(self, batch):
        # size = [224, 256, 288, 320, 352, 384, 416, 448, 480][np.random.randint(0, 9)]
        size = 256
        if self.cfg.is_transformer:
            size = 96
        if self.cfg.is_transformer_seg:
            size = 224

        image, mask, hcc_mask, cap_degree, cap_semantic, gt_c, name = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i], dsize=(size, size), interpolation=cv2.INTER_NEAREST)  # nearest or one-hot!
            cap_degree[i] = cv2.resize(cap_degree[i], dsize=(size, size), interpolation=cv2.INTER_NEAREST)
            cap_semantic[i] = cv2.resize(cap_semantic[i], dsize=(size, size), interpolation=cv2.INTER_NEAREST)
            hcc_mask[i] = cv2.resize(hcc_mask[i], dsize=(size, size), interpolation=cv2.INTER_NEAREST)
        # image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        image = torch.from_numpy(np.stack(image, axis=0)).unsqueeze(1)  # to tensor
        mask = torch.from_numpy(np.stack(mask, axis=0))  # .unsqueeze(1)
        gt_c = torch.from_numpy(np.stack(gt_c, axis=0)).unsqueeze(1)
        hcc_mask = torch.from_numpy(np.stack(hcc_mask, axis=0))

        cap_degree = torch.from_numpy(np.stack(cap_degree, axis=0))
        cap_semantic = torch.from_numpy(np.stack(cap_semantic, axis=0))

        return image, mask, hcc_mask, cap_degree, cap_semantic, gt_c, name

    def __len__(self):
        return len(self.samples)


########################### Testing Script ###########################
if __name__ == '__main__':
    pass
