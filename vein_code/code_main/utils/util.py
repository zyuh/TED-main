# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import torch
from torch.utils.data.sampler import Sampler
import cv2
import torch.nn as nn

def binary_dice_loss(x, y, require_sigmoid = True):
    if require_sigmoid:
        x = F.sigmoid(x)
    epislon = 1e-5
    batch_num = x.size(0)
    loss = 0
    for i in range(batch_num):
        intersect = torch.sum(x[i] * y[i])
        y_sum = torch.sum(y[i])
        x_sum = torch.sum(x[i])
        loss_ = 1 - ((2 * intersect + epislon) / (x_sum + y_sum + epislon))
        loss += loss_
    loss = loss / batch_num
    return loss


def dice_loss_1d(x, y, require_softmax = True):
    if require_softmax:
        x = nn.Softmax(dim=1)(x)
    epislon = 1e-5
    batch_num = x.size(0)
    loss = 0
    n_class=3
    for b in range(batch_num):
        for i in range(n_class):
            intersect = torch.sum(x[b][i] * (y[b]==i).float())
            y_sum = torch.sum(y[b]==i)
            x_sum = torch.sum(x[b][i])
            loss_ = 1 - ((2 * intersect + epislon) / (x_sum + y_sum + epislon))
            loss += loss_
    loss = loss / n_class / batch_num
    return loss

def gen_seqlabel_cartcoord_torch(cap_degree, cap_semantic, args=None):
    # can write to torch version, input (b, h, w)
    # assert torch.max(cap_degree)==90, "valid cap_degree range from 1 to 90"
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
                            

            d_semantic = cap_semantic[b, roi_nonzero[0][idx], roi_nonzero[1][idx]]
            seq_label[b, d] = d_semantic.item()
            # in case sampling the point with background semantics
            if d_semantic.item() == 0:
                if d > 0:
                    print("d>0 & containing seq_label=0 but removed~| name {}".format(args.name[b]))
                    seq_label[b, d] = seq_label[b, d-1]
                else:
                    print("d=0 & containing seq_label=0 and cannot be removed | name {}".format(args.name[b]))
                    pass

            vertexs[b, d, 0], vertexs[b, d, 1] = roi_nonzero[0][idx], roi_nonzero[1][idx]
        # assert torch.sum(seq_label[b]==0) == 0, "containing seq_label=0, name {}".format(args.name[b])

        if torch.sum(seq_label[b]==0) > 0:
            print("containing seq_label=0, name {}, sum {}".format(args.name[b], torch.sum(seq_label[b] == 0)))
            bins = torch.bincount(seq_label[b].int())
            argmax = torch.argmax(bins)
            seq_label[b, seq_label[b] == 0] = argmax.item()

    seq_label -= 1
    vertexs = (vertexs / 4.0).int()
    if 90 % args.n_ray != 0:
        assert 1==2, "90%args.n_ray!=0"

    return seq_label, vertexs


def vis(array, save_name, name_b, snapshot_path, vis_center=False, center=None, center_gt=None):
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
    cv2.imwrite(os.path.join(snapshot_path, name_b + '_' + save_name + '.png'), array_vis)


def _smooth_flatten(post_flatten, args=None):
    post_flatten_smooth = np.zeros((len(post_flatten), args.n_classes))
    for i in range(len(post_flatten)):
        for j in range(args.n_classes):
            post_flatten_smooth[i][j] = 1 if post_flatten[i][0] == j else 0
    post_flatten_smooth = torch.from_numpy(post_flatten_smooth).permute(1, 0).unsqueeze(0)
    post_flatten_smooth = nn.functional.interpolate(post_flatten_smooth, scale_factor=1 / 8,
                                                    mode='linear')
    post_flatten_smooth = nn.functional.interpolate(post_flatten_smooth, scale_factor=8,
                                                    mode='nearest').permute(2, 1, 0).repeat(1, 1,
                                                                                            args.ring_width)
    post_flatten_smooth = torch.argmax(post_flatten_smooth, dim=1).numpy()
    return post_flatten_smooth


def _recursive_search(idx, theta1_int, r1, theta1, len_window):
    # default len_window = 9
    neighbor = np.zeros((len_window, 2))
    half = int((len_window - 1) / 2)  # 4
    # print("start window..", len_window)
    for ii in range(len_window):
        idx_local = idx - half + ii
        if idx_local < 0:
            idx_local = 360 + idx_local
        elif idx_local > 360:
            idx_local = idx_local - 360
        tmp_r = np.array([r1[i] for i in np.where(theta1_int == np.float32(idx_local))])
        tmp_t = np.array([theta1[i] for i in np.where(theta1_int == np.float32(idx_local))])

        sortind = np.argsort(tmp_r)[0]
        tmp_r = np.array([tmp_r[0][id] for id in sortind])
        tmp_t = np.array([tmp_t[0][id] for id in sortind])  # almost the same number, not round

        tmp_r_len = len(tmp_r)
        # print('len(tmp_r)', len(tmp_r))
        if tmp_r_len != 0:
            tmp_r_mid, tmp_t_mid = tmp_r[int(len(tmp_r) / 2)], [int(len(tmp_t) / 2)]
            neighbor[ii, 0] = tmp_r_len
            neighbor[ii, 1] = tmp_r_mid
            # neighbor[ii, 1] = tmp_t_mid
        else:
            neighbor[ii, 0] = np.nan
            neighbor[ii, 1] = np.nan
    if np.all(np.isnan(neighbor)):
        # scale up the search scope! recursive searching?
        avg_window_width, avg_window_mid = _recursive_search(idx, theta1_int, r1, theta1, len_window=len_window + 2)
    else:
        avg_window_width = np.nanmean(neighbor[:, 0])
        avg_window_mid = np.nanmean(neighbor[:, 1])
    return avg_window_width, avg_window_mid


def _recursive_search_window(idx, post_flatten_tmp, init_len_window):
    half = int((init_len_window - 1) / 2)
    if idx < half:
        window = np.array(
            (post_flatten_tmp[idx - half:]).tolist() + (post_flatten_tmp[0:idx + half + 1]).tolist())
        # print('window', window)
    elif idx >= len(post_flatten_tmp) - half:
        window = np.array((post_flatten_tmp[idx - half:]).tolist() + (
            post_flatten_tmp[0:idx - len(post_flatten_tmp) + half + 1]).tolist())
    else:
        window = post_flatten_tmp[idx - half:idx + half + 1]
    if np.any(window > 0):  # and not (window[0] == 0 or window[-1]==0):
        # it is better to hit the new_id when both two end of the window is not zero., but results show vice verse
        counts = np.bincount(window)
        new_id = np.argmax(counts[1:]) + 1
        post_flatten_tmp[idx] = int(new_id)
        # print("satisfying", post_flatten_smooth)
    else:
        post_flatten_tmp = _recursive_search_window(idx, post_flatten_tmp, init_len_window + 2)
    return post_flatten_tmp


def _complete_flatten(post_flatten_tmp, r1, theta1,args=None):
    post_flatten_tmp = post_flatten_tmp[:, 0]
    theta1_int = [np.round(x) for x in theta1]
    for idxx in range(360):  # (0, 360)

        tmp = np.array([r1[i] for i in np.where(theta1_int == np.float32(idxx))])
        if len(tmp[0]) != 0 and post_flatten_tmp[idxx] != 0:
            continue
        # print("blank in slice ", idx)
        avg_window_width, avg_window_mid = _recursive_search(idxx, theta1_int, r1, theta1,
                                                             len_window=49)
        # print("post_flatten_smooth[idx]", idx, post_flatten_smooth[idx])
        if post_flatten_tmp[idxx] == 0:
            post_flatten_tmp = _recursive_search_window(idxx, post_flatten_tmp, init_len_window=9)
        r1 = np.concatenate((r1, np.arange(avg_window_mid - int(avg_window_width / 2),
                                           avg_window_mid + int(avg_window_width / 2) + 1)))
        len_new = 2 * int(avg_window_width / 2) + 1
        assert len(np.repeat(np.array([idxx]), len_new,
                             axis=0)) == len_new, "the new r1 and new theta1 have not same length"
        theta1 = np.concatenate((theta1, np.repeat(np.array([idxx]).astype(np.float), len_new, axis=0)))
    post_flatten_complete = np.repeat(np.reshape(post_flatten_tmp, (post_flatten_tmp.shape[0], 1)),
                                      args.ring_width, axis=1)
    return post_flatten_complete, r1, theta1


def post(tmp_stack, args=None):
    new_tmp_stack = []
    for i in range(len(tmp_stack)):
        counts = np.bincount(tmp_stack[i].astype(np.int))
        if len(counts) == 1:
            new_tmp_stack.append(0)
            continue
        new_tmp_stack.append(np.argmax(counts[1:]) + 1)
    new_tmp_stack = np.array(new_tmp_stack)
    new_tmp_stack = np.repeat(np.reshape(new_tmp_stack, (new_tmp_stack.shape[0], 1)), args.ring_width,
                              axis=1)
    return new_tmp_stack


def _recover_from_flatten(post_flatten, mask, M, r1, theta1):
    mask_recover = np.zeros_like(mask)
    cY, cX = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    theta1_int = [np.round(x) for x in theta1]
    for idx in range(360):
        tmp_r = np.array([r1[i] for i in np.where(theta1_int == np.float32(idx))])
        tmp_t = np.array([theta1[i] for i in np.where(theta1_int == np.float32(idx))])
        sortind = np.argsort(tmp_r)[0]
        tmp_r = [tmp_r[0][id] for id in sortind]
        tmp_t = [tmp_t[0][id] for id in sortind]  # the same numbers
        if len(tmp_r) == 0:  # filter !!!!
            continue
        tmp_r, tmp_t = np.array(tmp_r), np.array(tmp_t)
        x_new, y_new = cv2.polarToCart(tmp_r, tmp_t, angleInDegrees=True)
        x_cart_new, y_cart_new = x_new + cX, y_new + cY
        x_cart_new = [np.uint(np.round(x)) for x in x_cart_new]
        y_cart_new = [np.uint(np.round(y)) for y in y_cart_new]
        for i in range(len(y_cart_new)):
            mask_recover[x_cart_new[i], y_cart_new[i]] = post_flatten[idx][i]
    return mask_recover

def cart2polar(mask, M, cap, recover_flag=False, cap2=None, prgt_flag=False, args=None):
    if cap2 is not None:
        prgt_flag = True
        flatten_prgt = np.zeros((360, args.ring_width))
    # > pred mask (add), pred M
    # # > pred cap
    # # > gt cap: i.e. prgt_flag, results depend on pred mask
    # > gt mask, gt M
    # # > gt cap
    cart_x, cart_y = np.nonzero(mask)
    cart_x, cart_y = np.array(cart_x).astype(np.float32), np.array(cart_y).astype(np.float32)
    cY, cX = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    cart_offset_x, cart_offset_y = cart_x - cX, cart_y - cY
    r1, theta1 = cv2.cartToPolar(cart_offset_x, cart_offset_y, angleInDegrees=True)
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



def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[:self.N].astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[b] = sdf
            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf


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

