import numpy as np
from medpy import metric
import torch
from scipy.ndimage import zoom
from tqdm import tqdm


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return np.array([dice, hd95, asd])
    else:
        return np.zeros(3)


def cal_metric_binary(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return np.array([dice, hd95, asd])
    else:
        return np.zeros(3)


def cal_cap_metric(gt, pred, asd_flag=True):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        if asd_flag:
            asd = metric.binary.asd(pred, gt)
            return np.array([dice, asd])
        else:
            return np.array([dice])
    else:
        return np.zeros(2)


def cal_1d_dice(gt, pred):
    epsilon = 1e-15
    intersection = np.sum((gt * pred))
    union = np.sum(gt) + np.sum(pred)
    # gt=0, dice=0
    return (2 * intersection + epsilon) / (union+epsilon)


def cal_1d_dice_loader(gt, pred):
    epsilon = 1e-15
    intersection = np.sum((gt * pred))
    union = np.sum(gt) + np.sum(pred)

    return 2 * ((intersection + epsilon) / (union + epsilon))


def get_dice(y_true, y_pred):
    epsilon = 1e-15
    y_pred = (torch.sigmoid(y_pred) > 0.5).float()
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1) + epsilon

    return 2 * (intersection / union).mean()

def test_single_volume(image, label, net):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    # print('image', image.shape)
    # print('pred', prediction.shape)
    for ind in tqdm(range(image.shape[2])):
        slice = image[:, :, ind]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[:, :, ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    return first_metric, second_metric

