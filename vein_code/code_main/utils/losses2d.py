# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt as distance


def cross_entropy_weight_1d(x, y):
    # x_softmax = [nn.Softmax(1)(x[i]) for i in range(len(x))]
    # x_log = [np.log(x_softmax[i][y[i]]) for i in range(len(y))]
    # loss = - np.sum(x_log) / len(y)
    # return loss
    if len(x.size()) == 2:
        """x (b*360 3), y (b*360)  """
        n_batch, n_cls = x.size()
        # Normalizing to avoid instability
        x -= torch.max(x, dim=1, keepdim=True)
        # Compute Softmax activations
        probs = nn.Softmax(1)(x)
        logprobs = torch.zeros([n_batch, 1])
        for b in range(n_batch):  # loop batch
            scale_factor = 360 / float(torch.count_nonzero(y[b, :]))  # ?
            for c in range(n_cls):  # For each class
                if (y[b]==c).float() != 0:  # Positive classes
                    logprobs[b] += -torch.log(probs[b, c]) * (y[b]==c).float() * scale_factor  # We sum the loss per class for each element of the batch
        loss = torch.sum(logprobs) / len(x)
    if len(x.size()) == 3:
        """x (b, 3, 360), y (b, 360)  """
        n_batch, n_cls, n_width = x.size()
        # Normalizing to avoid instability
        x -= torch.max(x, dim=1, keepdim=True)
        # Compute Softmax activations
        probs = nn.Softmax(1)(x)
        logprobs = torch.zeros([n_batch, 1])
        for b in range(n_batch):  # loop batch
            # scale_factor = n_width / float(torch.count_nonzero(y[b]))  # ?
            for c in range(n_cls):  # For each class
                if (y[b]==c).float() != 0:  # Positive classes
                    logprobs[b] += -torch.log(probs[b, c, :]) * (y[b]==c).float() #  * scale_factor  # We sum the loss per class for each element of the batch
        loss = torch.sum(logprobs) / len(x)

def get_soft_label(input_tensor, num_class):
    """
        convert a label tensor to soft label
        input_tensor: tensor with shae [B, 1, D, H, W]
        output_tensor: shape [B, num_class, D, H, W]
    """
    tensor_list = []
    for i in range(num_class):
        temp_prob = input_tensor == i*torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    output_tensor = output_tensor.double()

    return output_tensor


def reshape_prediction_and_ground_truth(predict, soft_y):
    """
    reshape input variables of shape [B, C, D, H, W] to [voxel_n, C]
    """
    tensor_dim = len(predict.size())
    num_class = list(predict.size())[1]
    if(tensor_dim == 5):
        soft_y = soft_y.permute(0, 2, 3, 4, 1)
        predict = predict.permute(0, 2, 3, 4, 1)
    elif(tensor_dim == 4):
        soft_y = soft_y.permute(0, 2, 3, 1)
        predict = predict.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))

    predict = torch.reshape(predict, (-1, num_class))
    soft_y = torch.reshape(soft_y,  (-1, num_class))

    return predict, soft_y


def cross_entropy_loss(predict, soft_y, softmax=True):
    """
    get cross entropy scores for each class in predict and soft_y
    """
    if(softmax):
        predict = nn.Softmax(dim=1)(predict)
    predict, soft_y = reshape_prediction_and_ground_truth(predict, soft_y)

    ce = - soft_y * torch.log(predict)
    ce = torch.mean(ce, dim=0)
    ce = torch.sum(ce)
    return ce


def get_classwise_dice(predict, soft_y):
    """
    get dice scores for each class in predict (after softmax) and soft_y
    """
    y_vol = torch.sum(soft_y,  dim=0)
    p_vol = torch.sum(predict, dim=0)
    intersect = torch.sum(soft_y * predict, dim=0)
    dice_score = (2.0 * intersect + 1e-5) / (y_vol + p_vol + 1e-5)
    return dice_score


def soft_dice_loss(predict, soft_y, softmax=True):
    if(softmax):
        predict = nn.Softmax(dim=1)(predict)
    predict, soft_y = reshape_prediction_and_ground_truth(predict, soft_y)
    dice_score = get_classwise_dice(predict, soft_y)
    dice_loss = 1.0 - torch.mean(dice_score)
    return dice_loss


def ce_dice_loss(predict, soft_y, softmax=True):
    if(softmax):
        predict = nn.Softmax(dim=1)(predict)
    predict, soft_y = reshape_prediction_and_ground_truth(predict, soft_y)

    ce = - soft_y * torch.log(predict)
    ce = torch.mean(ce, dim=0)
    ce = torch.sum(ce)

    dice_score = get_classwise_dice(predict, soft_y)
    dice_loss = 1.0 - torch.mean(dice_score)

    loss = ce + dice_loss
    return loss


def volume_dice_loss(predict, soft_y, softmax=True):
    if(softmax):
        predict = nn.Softmax(dim=1)(predict)
    predict, soft_y = reshape_prediction_and_ground_truth(predict, soft_y)
    dice_score = get_classwise_dice(predict, soft_y)
    dice_loss = 1.0 - torch.mean(dice_score)

    vp = torch.sum(predict, dim=0)
    vy = torch.sum(predict, dim=0)
    v_loss = (vp - vy)/vy
    v_loss = v_loss * v_loss
    v_loss = torch.mean(v_loss)

    loss = dice_loss + v_loss * 0.2
    return loss


def hardness_weight_dice_loss(predict, soft_y, softmax=True):
    if(softmax):
        predict = nn.Softmax(dim=1)(predict)
    predict, soft_y = reshape_prediction_and_ground_truth(predict, soft_y)

    weight = torch.abs(predict - soft_y)
    lamb = 0.6
    weight = lamb + weight*(1 - lamb)

    y_vol = torch.sum(soft_y*weight,  dim=0)
    p_vol = torch.sum(predict*weight, dim=0)
    intersect = torch.sum(soft_y * predict * weight, dim=0)
    dice_score = (2.0 * intersect + 1e-5) / (y_vol + p_vol + 1e-5)
    dice_loss = 1.0 - torch.mean(dice_score)
    return dice_loss


def exponentialized_dice_loss(predict, soft_y, softmax=True):
    if(softmax):
        predict = nn.Softmax(dim=1)(predict)
    dice_score = get_classwise_dice(predict, soft_y)
    exp_dice = - torch.log(dice_score)
    exp_dice = torch.mean(exp_dice)
    return exp_dice


def generalized_dice_loss(predict, soft_y, softmax=True):
    tensor_dim = len(predict.size())
    num_class = list(predict.size())[1]
    if(softmax):
        predict = nn.Softmax(dim=1)(predict)
    if(tensor_dim == 5):
        soft_y = soft_y.permute(0, 2, 3, 4, 1)
        predict = predict.permute(0, 2, 3, 4, 1)
    elif(tensor_dim == 4):
        soft_y = soft_y.permute(0, 2, 3, 1)
        predict = predict.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))

    soft_y = torch.reshape(soft_y,  (-1, num_class))
    predict = torch.reshape(predict, (-1, num_class))
    num_voxel = list(soft_y.size())[0]
    vol = torch.sum(soft_y, dim=0)
    weight = (num_voxel - vol)/num_voxel
    intersect = torch.sum(predict * soft_y, dim=0)
    intersect = torch.sum(weight * intersect)
    vol_sum = torch.sum(soft_y, dim=0) + torch.sum(predict, dim=0)
    vol_sum = torch.sum(weight * vol_sum)
    dice_score = (2.0 * intersect + 1e-5) / (vol_sum + 1e-5)
    dice_loss = 1.0 - dice_score
    return dice_loss


def distance_loss(predict, soft_y, lab_distance, softmax=True):
    """
    get distance loss function
    lab_distance is unsigned distance transform of foreground contour
    """
    tensor_dim = len(predict.size())
    num_class = list(predict.size())[1]
    if(softmax):
        predict = nn.Softmax(dim=1)(predict)
    if(tensor_dim == 5):
        lab_distance = lab_distance.permute(0, 2, 3, 4, 1)
        predict = predict.permute(0, 2, 3, 4, 1)
        soft_y = soft_y.permute(0, 2, 3, 4, 1)
    elif(tensor_dim == 4):
        lab_distance = lab_distance.permute(0, 2, 3, 1)
        predict = predict.permute(0, 2, 3, 1)
        soft_y = soft_y.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))

    lab_distance = torch.reshape(lab_distance,  (-1, num_class))
    predict = torch.reshape(predict, (-1, num_class))
    soft_y = torch.reshape(soft_y, (-1, num_class))

    # mis_seg  = torch.abs(predict - soft_y)
    dis_sum = torch.sum(lab_distance * predict, dim=0)
    vox_sum = torch.sum(predict, dim=0)
    avg_dis = (dis_sum + 1e-5)/(vox_sum + 1e-5)
    avg_dis = torch.mean(avg_dis)
    return avg_dis


def dice_distance_loss(predict, soft_y, lab_distance, softmax=True):
    dice_loss = soft_dice_loss(predict, soft_y, softmax)
    dis_loss = distance_loss(predict, soft_y, lab_distance, softmax)
    loss = dice_loss + 0.2 * dis_loss
    return loss


def Active_Contour_Loss_org(y_true, y_pred):

    # horizontal and vertical directions
    x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    y = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]

    delta_x = x[:, :, 1:, :-2]**2
    delta_y = y[:, :, :-2, 1:]**2
    delta_u = torch.abs(delta_x + delta_y)

    # where is a parameter to avoid square root is zero in practice.
    epsilon = 1e-8
    w = 1
    # eq.(11) in the paper.
    lenth = w * torch.sum(torch.sqrt(delta_u + epsilon))

    """region term"""
    C_1 = torch.ones_like(y_pred)
    C_2 = torch.zeros_like(y_pred)

    # equ.(12) in the paper.
    region_in = torch.abs(torch.sum(y_pred * ((y_true - C_1)**2)))
    # equ.(12) in the paper.
    region_out = torch.abs(torch.sum((1-y_pred) * ((y_true - C_2)**2)))
    lambdaP = 1  # lambda parameter could be various.
    loss = lenth + lambdaP * (region_in + region_out)

    return loss


def Active_Contour_Loss(y_pred, y_true):
    y_pred = y_pred.float()
    # horizontal and vertical directions
    x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    y = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]

    delta_x = x[:, :, 1:, :-2]**2
    delta_y = y[:, :, :-2, 1:]**2
    delta_u = torch.abs(delta_x + delta_y)

    # where is a parameter to avoid square root is zero in practice.
    epsilon = 1e-8
    w = 1
    # eq.(11) in the paper.
    lenth = w * torch.mean(torch.sqrt(delta_u + epsilon))

    """region term"""
    C_1 = torch.ones_like(y_pred)
    C_2 = torch.zeros_like(y_pred)
    y_true = y_true.float()
    # equ.(12) in the paper.
    region_in = torch.abs(torch.mean(y_pred * ((y_true - C_1)**2)))
    # equ.(12) in the paper.
    region_out = torch.abs(torch.mean((1-y_pred) * ((y_true - C_2)**2)))
    lambdaP = 1  # lambda parameter could be various.
    loss = lenth + lambdaP * (region_in + region_out)

    return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        # has done label.unsqueeze(1)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        # also suitable for 1D
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        # target: tensor with shape[B, 1, D, H, W]
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class levelsetLoss(nn.Module):
    def __init__(self):
        super(levelsetLoss, self).__init__()

    def forward(self, output, target):
        # input size = batch x 1 (channel) x height x width
        outshape = output.shape
        tarshape = target.shape
        loss = 0.0
        for ich in range(tarshape[1]):
            target_ = torch.unsqueeze(target[:, ich], 1)
            target_ = target_.expand(
                tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pcentroid = torch.sum(target_ * output, (2, 3)) / \
                torch.sum(output, (2, 3))
            pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
            plevel = target_ - \
                pcentroid.expand(
                    tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pLoss = plevel * plevel * output
            loss += torch.sum(pLoss)
        return loss


def loss_zoos(loss="dice"):
    if loss == "dice":
        criterion = DiceLoss(3)
    if loss == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    if loss == "ac":
        criterion = Active_Contour_Loss
    if loss == "levelset":
        criterion = levelsetLoss()
    return criterion


def entropy_minmization(p, c=4):
    p = torch.softmax(p, dim=1)
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(c)).cuda()
    ent = torch.mean(y1)
    return ent


def entropy_map(p, c=4):
    p = torch.softmax(p, dim=1)
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True) / torch.tensor(np.log(c)).cuda()
    return ent_map
