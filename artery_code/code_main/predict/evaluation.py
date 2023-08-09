import numpy as np
from numpy.core.fromnumeric import mean
import os
import SimpleITK as sitk
import copy


class runningScore(object):
    def __init__(self, args):
        self.args = args
        self.n_classes = self.args.num_classes
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2, ).reshape(n_class, n_class)
        return hist


    def update(self, label_trues, label_preds, step=-1):
        lt, lp = label_trues, label_preds
        img_lt = sitk.ReadImage(lt)
        img_lt_npy = sitk.GetArrayFromImage(img_lt)

        img_lp = sitk.ReadImage(lp)
        img_lp_npy = sitk.GetArrayFromImage(img_lp)
        print(label_trues, label_preds)
        print(img_lt_npy.shape, img_lp_npy.shape)
        self.confusion_matrix += self._fast_hist(img_lt_npy.flatten(), img_lp_npy.flatten(), self.n_classes)


    def get_scores(self):
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        prec = acc_cls.copy()
        acc_cls = np.nanmean(acc_cls)
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iou)
        cls_iou = dict(zip(range(self.n_classes), iou))
        dice = 2 * np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0))
        cls_dice = dict(zip(range(self.n_classes), dice))
        recall = np.diag(hist) / hist.sum(axis=0)
        mean_recall = np.nanmean(recall)
        f1score = 2/(1/acc_cls + 1/mean_recall)
        f1 = 2/(1/prec + 1/recall)

        cls_f1 = dict(zip(range(self.n_classes), f1))
        return (
            {
                "Overall Acc": acc,
                "Mean Prec": acc_cls,
                "Mean IoU": mean_iou,
                "Mean Recall": mean_recall,
                "Mean f1score": f1score,
                "Class IoU": cls_iou,
            },
            {
                "Class Dice": dice,
                "confusion_matrix": self.confusion_matrix
            }
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


if __name__=='__main__':
    pass
