import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.triplet_mining import online_mine_all, online_mine_hard


class TripletLoss(object):
  """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid). 
  Related Triplet Loss theory can be found in paper 'In Defense of the Triplet 
  Loss for Person Re-Identification'."""
  def __init__(self, margin=None):
    self.margin = margin
    if margin is not None:
      self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    else:
      self.ranking_loss = nn.SoftMarginLoss()

  def __call__(self, dist_ap, dist_an):
    """
    Args:
      dist_ap: pytorch Variable, distance between anchor and positive sample, 
        shape [N]
      dist_an: pytorch Variable, distance between anchor and negative sample, 
        shape [N]
    Returns:
      loss: pytorch Variable, with shape [1]
    """
    y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
    if self.margin is not None:
      loss = self.ranking_loss(dist_an, dist_ap, y)
    else:
      loss = self.ranking_loss(dist_an - dist_ap, y)
    return loss


class Mlp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mlp, self).__init__()
        # only deal with last dimension !!! https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc1 = nn.Linear(in_channels, 64)
        self.fc2 = nn.Linear(64, out_channels)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        # x = x.permute(0, 2, 1)
        return x


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400,
                 args=None):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.args = args

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels, self.in_planes, kernel_size=(conv1_t_size, 7, 7), stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout2d(0.5)

        if self.args.use_bio_marker:
            self.fc = Mlp(in_channels=(block_inplanes[3] * block.expansion +7 +1), out_channels=n_classes)
        else:
            self.fc = Mlp(in_channels=(block_inplanes[3] * block.expansion +1), out_channels=n_classes)



        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out


    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, resize_factor, cap1, cap2, cap3, fen1, fen2, fen3, num, radio, have_ace_or_not):
        resize_factor = torch.tensor(resize_factor).unsqueeze(1).cuda().float()

        if self.args.use_bio_marker:
            cap1 = torch.tensor(cap1).unsqueeze(1).cuda().float()
            cap2 = torch.tensor(cap2).unsqueeze(1).cuda().float()
            fen1 = torch.tensor(fen1).unsqueeze(1).cuda().float()
            fen2 = torch.tensor(fen2).unsqueeze(1).cuda().float()
            num = torch.tensor(num).unsqueeze(1).cuda().float()
            radio = torch.tensor(radio).unsqueeze(1).cuda().float()
            have_ace_or_not = torch.tensor(have_ace_or_not).unsqueeze(1).cuda().float()

        each_logist_data = self.once_forward(x)
        self.embeddings = each_logist_data

        if self.args.use_bio_marker:
            each_logist_data_plus = torch.cat([each_logist_data, cap1, cap2, fen1, fen2], dim=1)
            each_logist_data_plus = torch.cat([each_logist_data_plus, num, radio, have_ace_or_not], dim=1)
            logists = torch.cat([each_logist_data_plus, resize_factor], dim=1)
        else:
            logists = torch.cat([each_logist_data, resize_factor], dim=1)    

        logists = self.dropout(logists)
        output = self.fc(logists)

        return output
            

    def once_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        return x
    


    def get_triplet_loss(self, labels, margin, device='cpu', type='batch_all'):# 暂不考虑三组，先考虑两组
        if type == 'batch_all':
            triplet_loss, num_positive_triplets, num_valid_triplets = online_mine_all(labels, self.embeddings, margin, squared=False, device=device)
        else:
            assert type == 'batch_hard'
            triplet_loss, mask_anchor_positive, mask_anchor_negative = online_mine_hard(labels, self.embeddings, margin, squared=False, device=device)

        return triplet_loss


    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name):
        torch.save(self.state_dict(), name)
        print('Model ' + name + ' has been saved!')
        return name




def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

