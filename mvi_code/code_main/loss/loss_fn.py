import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class LabelSmoothing(nn.Module):
    """Implement label smoothing.  size表示类别总数"""

    def __init__(self, size, smoothing=0.0, reduction='batchmean'):
        super(LabelSmoothing, self).__init__()

        self.criterion = nn.KLDivLoss(size_average=False, reduction=reduction)

        # self.padding_idx = padding_idx

        self.confidence = 1.0 - smoothing

        self.smoothing = smoothing

        self.size = size

        self.true_dist = None

    def forward(self, x, target):
        """
        x表示输入 (N,M)N个样本,M表示总类数,每一个类的概率log P
        target表示label(M)
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()  # 先深复制过来
        # print(true_dist)
        true_dist.fill_(self.smoothing / (self.size - 1))  # otherwise的公式
        # print(true_dist)

        # 变成one-hot编码，1表示按列填充，
        # target.data.unsqueeze(1)表示索引,confidence表示填充的数字
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        true_dist.requires_grad = False
        self.true_dist = true_dist

        return self.criterion(x, true_dist)


def minentropyloss(logits):
    logits_log_softmax = torch.log_softmax(logits, dim=1)
    num_classed = logits.size()[1]
    entropy = -torch.softmax(logits, dim=1)  * logits_log_softmax
    entropy_reg = torch.mean(entropy)
    return entropy_reg