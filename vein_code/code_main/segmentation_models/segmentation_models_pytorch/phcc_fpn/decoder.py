import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class ChReduce_Block(nn.Module):
    def __init__(self, pyramid_channels, out_channels):
        """newly added"""
        super().__init__()
        self.conv1 = nn.Conv2d(pyramid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy)
            )

class AddCoordsTh(nn.Module):
    def __init__(self, with_r=False):
        super(AddCoordsTh, self).__init__()
        self.x_dim = 0
        self.y_dim = 0
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        input_tensor: (batch, c, x_dim, y_dim)
        """

        batch_size_tensor, _, self.x_dim, self.y_dim = input_tensor.shape

        xx_ones = torch.ones([1, self.y_dim], dtype=torch.int32).cuda()
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = torch.arange(self.x_dim, dtype=torch.int32).unsqueeze(0).cuda()
        xx_range = xx_range.unsqueeze(1)

        xx_channel = torch.matmul(xx_ones.float(), xx_range.float())
        xx_channel = xx_channel.unsqueeze(-1)

        yy_ones = torch.ones([1, self.x_dim], dtype=torch.int32).cuda()
        yy_ones = yy_ones.unsqueeze(1)

        yy_range = torch.arange(self.y_dim, dtype=torch.int32).unsqueeze(0).cuda()
        yy_range = yy_range.unsqueeze(-1)

        yy_channel = torch.matmul(yy_range.float(), yy_ones.float())
        yy_channel = yy_channel.unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 2, 1)
        yy_channel = yy_channel.permute(0, 3, 2, 1)

        xx_channel = xx_channel.float() / (self.x_dim - 1)
        yy_channel = yy_channel.float() / (self.y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConvTh(nn.Module):
    """CoordConv layer as in the paper."""
    def __init__(self, with_r, *args, **kwargs):
        super(CoordConvTh, self).__init__()

        self.addcoords = AddCoordsTh(with_r=with_r)
        # self.conv = nn.Conv2d(*args, **kwargs)
        self.conv = Conv3x3GNReLU(256+2, 256+2)

    def forward(self, input_tensor):
        ret = self.addcoords(input_tensor)
        # ret = self.conv(ret)
        return ret


class vertex_sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, vertexs):
        """
        :param x: (b, c, h/4, w/4)
        :param vertexs: (b, 360/4, 2), polar vertex coordinates in 1/4 images
        :return: (b, c, 360/4)
        """
        B, Ch, h4, w4 = x.size()
        n_ray = vertexs.size()[1]
        out = torch.zeros((B, Ch, n_ray), device=x.device)
        for b in range(B):
            for idx in range(n_ray):
                i, j = vertexs[b, idx]
                out[b, :, idx] = x[b, :, i, j]
        # ret = self.conv(ret)
        return out



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
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x

class FPNDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            encoder_depth=5,
            pyramid_channels=256,
            segmentation_channels=128,
            dropout=0.2,
            merge_policy="add",
            is_tumor_seg=False,
            is_no_fpn=False,
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[:encoder_depth + 1]

        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.cr5 = ChReduce_Block(pyramid_channels, 64)
        self.cr4 = ChReduce_Block(pyramid_channels, 64)
        self.cr3 = ChReduce_Block(pyramid_channels, 64)
        self.cr2 = ChReduce_Block(pyramid_channels, 64)

        self.coorconv = CoordConvTh(with_r=False)

        self.up_blocks = nn.ModuleList([
            nn.UpsamplingBilinear2d(scale_factor=scale)
            for scale in [8, 4, 2, 1]
        ])

        self.is_tumor_seg = is_tumor_seg
        self.is_no_fpn = is_no_fpn
        if self.is_tumor_seg:
            self.seg_blocks = nn.ModuleList([
                SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
                for n_upsamples in [3, 2, 1, 0]
            ])
            self.merge_tumor = MergeBlock('add')
            self.dropout_tumor = nn.Dropout2d(p=dropout, inplace=True)

        # self.merge = MergeBlock(merge_policy)
        self.merge = MergeBlock('cat')
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)


    def forward(self, *features):
        c2, c3, c4, c5 = features[-4:]
        # print("c2-25", c2.size(), c3.size(), c4.size(), c5.size())

        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)
        if self.is_tumor_seg:
            feature_pyramid_tumor = [seg_block(p.clone()) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
            mask_tumor = self.merge_tumor(feature_pyramid_tumor)
            mask_tumor = self.dropout_tumor(mask_tumor)

        if self.is_no_fpn:
            x = p2
        else:
            # reduce pyramid channel to 64
            p5 = self.cr5(p5)
            p4 = self.cr4(p4)
            p3 = self.cr3(p3)
            p2 = self.cr2(p2)
            # feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
            feature_pyramid = [up_block(p) for up_block, p in zip(self.up_blocks, [p5, p4, p3, p2])]
            x = self.merge(feature_pyramid)  # 64*4=256

        x = self.dropout(x)

        if self.is_tumor_seg:
            return x, mask_tumor

        return x
