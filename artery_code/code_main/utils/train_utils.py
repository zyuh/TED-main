import os
import sys
import time
from datetime import datetime
import shutil
import numpy as np
from visdom import Visdom
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def load_MA(Basic_Args, all_tr_losses, all_val_losses):
    length = len(all_tr_losses[0])
    train_loss_MA = all_tr_losses[0][0]
    val_eval_criterion_MA = all_val_losses[0][0]
    for i in range(length):
        train_loss_MA = Basic_Args.train_loss_MA_alpha * train_loss_MA + \
                        (1 - Basic_Args.train_loss_MA_alpha) * all_tr_losses[0][i]
        val_eval_criterion_MA = Basic_Args.val_eval_criterion_alpha * val_eval_criterion_MA + \
                                (1 - Basic_Args.val_eval_criterion_alpha) * all_val_losses[0][i]
    
    return train_loss_MA, val_eval_criterion_MA


def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=True)
    return data


def run_online_evaluation(output, target):
    with torch.no_grad():
        num_classes = output.shape[1]
        softmax_helper = lambda x: F.softmax(x, 1)
        output_softmax = softmax_helper(output)
        output_seg = output_softmax.argmax(1)
        target = target[:, 0]
        axes = tuple(range(1, len(target.shape)))
        tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
        fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
        fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
        for c in range(1, num_classes):
            tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
            fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
            fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

        tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

        return list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)), list(tp_hard), list(fp_hard), list(fn_hard)


def finish_online_evaluation(online_eval_tp, online_eval_fp, online_eval_fn):
    online_eval_tp = np.sum(online_eval_tp, 0)
    online_eval_fp = np.sum(online_eval_fp, 0)
    online_eval_fn = np.sum(online_eval_fn, 0)

    global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                        zip(online_eval_tp, online_eval_fp, online_eval_fn)] if not np.isnan(i)]

    return global_dc_per_class

