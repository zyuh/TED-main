from typing import Optional, Union
from .decoder import FPNDecoder
from ..base import  SegmentationModel, SegmentationHead, ClassificationHead
from ..base import initialization as init
from ..encoders import get_encoder
from .decoder import CoordConvTh, vertex_sampler, Mlp
import torch
import torch.nn as nn
import numpy as np
import cv2

class PHCC_FPN(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.1,  # default 0.2
        in_channels: int = 3,
        classes: int = 1,
        attention: Optional[str] = None,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
        is_tumor_seg: bool = False,
        is_testvertex: bool = False,
        is_no_coorconv: bool = False,
        is_no_fpn: bool = False,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
            is_tumor_seg=is_tumor_seg,
            is_no_fpn=is_no_fpn,
        )
        self.is_tumor_seg = is_tumor_seg
        self.is_testvertex = is_testvertex
        self.is_no_coorconv = is_no_coorconv
        if self.is_tumor_seg:
            self.segmentation_head = SegmentationHead(
                in_channels=self.decoder.out_channels,
                out_channels=2,
                activation=activation,
                kernel_size=1,
                upsampling=upsampling,
            )

        self.attention_type = attention

        if self.attention_type == 'transformer':
            from .transformer import Transformer, CONFIGS
            CONFIGS['ViT-tiny'].embedding_in_channels = 256 if self.is_no_coorconv else 258
            self.attention = Transformer(CONFIGS['ViT-tiny'])
            mlp_inchannel = CONFIGS['ViT-tiny'].hidden_size
        else:
            mlp_inchannel = 256 if self.is_no_coorconv else 258

        # self.coorconv = CoordConvTh(x_dim=int(224/4), y_dim=int(224/4), with_r=False)
        if not self.is_no_coorconv:
            self.coorconv = CoordConvTh(with_r=False)
        self.vertex_sampler = vertex_sampler()

        self.mlp = Mlp(in_channels=mlp_inchannel, out_channels=classes-1)

        self.name = "fpn-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        # init.initialize_head(self.segmentation_head)

    def forward(self, x, vertexs):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        b = x.shape[0]

        if self.is_tumor_seg:
            feat_dec, feat_tumor = self.decoder(*features)  # (b, 256, h/4, w/4)
            masks = self.segmentation_head(feat_tumor)  # (b, 4, h, w)
            if self.is_testvertex:  # vertexs is None:
                mask_tumor = torch.argmax(torch.softmax(masks, dim=1), dim=1).cpu().detach().numpy()  # (b, h, w)
                # convert mask_tumor into vertexs  (B, 90, 2)
                vertexs_new = torch.zeros_like(vertexs) # torch.zeros((len(mask_tumor), 90, 2))
                for b in range(len(mask_tumor)):
                    label_hcc_b = mask_tumor[b]
                    dilation = cv2.dilate(label_hcc_b.astype(np.uint8), kernel=np.ones((3, 3), np.uint8), iterations=1)
                    erosion = cv2.erode(label_hcc_b.astype(np.uint8), kernel=np.ones((3, 3), np.uint8), iterations=1)
                    hcc_cvt_edge_b = (dilation - erosion).astype(np.float)
                    hcc_cvt_edge_b = (cv2.GaussianBlur(hcc_cvt_edge_b, (5, 5), 0) > 0).astype(np.float)  # default (3, 3)
                    M_cvt_gt = cv2.moments(hcc_cvt_edge_b)  # to make online recover easier! replace hcc_cvt_edge_b
                    if np.sum(hcc_cvt_edge_b) == 0:  # data to be clean
                        vertexs_new[b] = vertexs[b] * 4
                        print("--------------------------- noisy data", b)
                        continue
                    cY_cvt_gt, cX_cvt_gt = int(M_cvt_gt["m10"] / M_cvt_gt["m00"]), int(M_cvt_gt["m01"] / M_cvt_gt["m00"])
                    flatten_cvt_gt, cvt_r1, cvt_theta1 = cart2polar(hcc_cvt_edge_b, M_cvt_gt, hcc_cvt_edge_b,recover_flag=True)
                    theta1_int = [np.round(x) for x in cvt_theta1]
                    check_complete = np.array(
                        [len(np.array([cvt_r1[i] for i in np.where(theta1_int == np.float32(idx))])[0]) == 0 for idx in
                         range(360)])
                    if np.any(check_complete):
                        # print("name {} how many {}".format(name_b, np.sum(check_complete)))
                        post_flatten_cvt_gt = post(flatten_cvt_gt)
                        post_flatten_cvt_gt = _smooth_flatten(post_flatten_cvt_gt)
                        post_flatten_cvt_gt, cvt_r1, cvt_theta1 = _complete_flatten(post_flatten_cvt_gt.copy(), cvt_r1,
                                                                                    cvt_theta1)
                        theta1_int = [np.round(x) for x in cvt_theta1]

                    for idx_scale4 in range(90):
                        idx = idx_scale4 * 4
                        x_cart_new, y_cart_new = degree2cart(idx, None, cvt_r1, cvt_theta1, theta1_int, cX_cvt_gt, cY_cvt_gt)
                        i_mid = int(len(x_cart_new) / 2)
                        vertexs_new[b][idx_scale4][0] = int(x_cart_new[i_mid])
                        vertexs_new[b][idx_scale4][1] = int(y_cart_new[i_mid])
                vertexs = vertexs_new
                vertexs = (vertexs / 4.0).int()
        else:
            feat_dec = self.decoder(*features)  # (b, 256, h/4, w/4)

        if not self.is_no_coorconv:
            feat_dec = self.coorconv(feat_dec)  # (b, 256+2, h/4, w/4)

        feat_seq = self.vertex_sampler(feat_dec, vertexs)  # (b, 256+2, 360/4)
        if self.attention_type == 'transformer':
            feat_seq = self.attention(feat_seq)  # (b, hidden, 360/4)

        out_seq = self.mlp(feat_seq)  # (b, n_class, 360/4)

        if self.is_tumor_seg:
            return out_seq, masks
        return out_seq

    def predict(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x



def degree2cart(idx, name_b, cvt_r1, cvt_theta1, theta1_int, cX_cvt_gt, cY_cvt_gt):
    tmp_r = np.array([cvt_r1[i] for i in np.where(theta1_int == np.float32(idx))])
    tmp_t = np.array([cvt_theta1[i] for i in np.where(theta1_int == np.float32(idx))])
    sortind = np.argsort(tmp_r)[0]
    tmp_r = [tmp_r[0][id] for id in sortind]
    tmp_t = [tmp_t[0][id] for id in sortind]
    tmp_r, tmp_t = np.array(tmp_r), np.array(tmp_t)
    if len(tmp_r) == 0:  # filter !!!!
        print("the boundary is not complete, degree {} ".format(idx % 4))
        assert 1==2
        # return None, None
    x_new, y_new = cv2.polarToCart(tmp_r, tmp_t, angleInDegrees=True)
    x_cart_new, y_cart_new = x_new + cX_cvt_gt, y_new + cY_cvt_gt
    x_cart_new = [np.uint(np.round(x)) for x in x_cart_new]
    y_cart_new = [np.uint(np.round(y)) for y in y_cart_new]
    i_mid = int(len(x_cart_new) / 2)
    return x_cart_new, y_cart_new


def cart2polar(mask, M, cap, recover_flag=False, cap2=None, prgt_flag=False):
    if cap2 is not None:
        prgt_flag = True
        flatten_prgt = np.zeros((360, 150))
    cart_x, cart_y = np.nonzero(mask)
    cart_x, cart_y = np.array(cart_x).astype(np.float32), np.array(cart_y).astype(np.float32)
    cY, cX = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    cart_offset_x, cart_offset_y = cart_x - cX, cart_y - cY
    r1, theta1 = cv2.cartToPolar(cart_offset_x, cart_offset_y, angleInDegrees=True)
    r1, theta1 = np.array(r1)[:, 0], np.array(theta1)[:, 0]
    theta1_int = [np.round(x) for x in theta1]
    count_len = []
    flatten = np.zeros((360, 150)) # np.zeros((360, args.ring_width))
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


def _smooth_flatten(post_flatten):
    n_classes = 4
    ring_width=150
    post_flatten_smooth = np.zeros((len(post_flatten), n_classes))
    for i in range(len(post_flatten)):
        for j in range(n_classes):
            post_flatten_smooth[i][j] = 1 if post_flatten[i][0] == j else 0
    post_flatten_smooth = torch.from_numpy(post_flatten_smooth).permute(1, 0).unsqueeze(0)
    post_flatten_smooth = nn.functional.interpolate(post_flatten_smooth, scale_factor=1 / 8,
                                                    mode='linear')
    post_flatten_smooth = nn.functional.interpolate(post_flatten_smooth, scale_factor=8,
                                                    mode='nearest').permute(2, 1, 0).repeat(1, 1,ring_width)
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


def _complete_flatten(post_flatten_tmp, r1, theta1):
    ring_width=150
    post_flatten_tmp = post_flatten_tmp[:, 0]
    theta1_int = [np.round(x) for x in theta1]
    for idxx in range(360):  # (0, 360)

        tmp = np.array([r1[i] for i in np.where(theta1_int == np.float32(idxx))])
        if len(tmp[0]) != 0 and post_flatten_tmp[idxx] != 0:
            continue
        # print("blank in slice ", idx)
        avg_window_width, avg_window_mid = _recursive_search(idxx, theta1_int, r1, theta1,
                                                             len_window=49)

        if post_flatten_tmp[idxx] == 0:
            post_flatten_tmp = _recursive_search_window(idxx, post_flatten_tmp, init_len_window=9)
        r1 = np.concatenate((r1, np.arange(avg_window_mid - int(avg_window_width / 2),
                                           avg_window_mid + int(avg_window_width / 2) + 1)))
        len_new = 2 * int(avg_window_width / 2) + 1
        assert len(np.repeat(np.array([idxx]), len_new,
                             axis=0)) == len_new, "the new r1 and new theta1 have not same length"
        theta1 = np.concatenate((theta1, np.repeat(np.array([idxx]).astype(np.float), len_new, axis=0)))
    post_flatten_complete = np.repeat(np.reshape(post_flatten_tmp, (post_flatten_tmp.shape[0], 1)),
                                      ring_width, axis=1)
    return post_flatten_complete, r1, theta1


def post(tmp_stack):
    ring_width=150
    new_tmp_stack = []
    for i in range(len(tmp_stack)):
        counts = np.bincount(tmp_stack[i].astype(np.int))
        if len(counts) == 1:
            new_tmp_stack.append(0)
            continue
        new_tmp_stack.append(np.argmax(counts[1:]) + 1)
    new_tmp_stack = np.array(new_tmp_stack)
    new_tmp_stack = np.repeat(np.reshape(new_tmp_stack, (new_tmp_stack.shape[0], 1)), ring_width,
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