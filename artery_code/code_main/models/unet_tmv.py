from copy import deepcopy
from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from typing import Union, Tuple, List

from torch.cuda.amp import autocast
from utils.file_ops import * 
from utils.train_utils import *
import copy

softmax_helper = lambda x: F.softmax(x, 1)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def get_device(self):
        if next(self.parameters()).device == "cpu":
            return "cpu"
        else:
            return next(self.parameters()).device.index

    def set_device(self, device):
        if device == "cpu":
            self.cpu()
        else:
            self.cuda(device)

    def forward(self, x):
        raise NotImplementedError


class SegmentationNetwork(NeuralNetwork):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.input_shape_must_be_divisible_by = None
        self.conv_op = None
        self.num_classes = None
        self._gaussian_3d = self._patch_size_for_gaussian_3d = None
        self._gaussian_2d = self._patch_size_for_gaussian_2d = None

    def predict_3D(self, x, do_mirroring, mirror_axes, step_size, 
                   patch_size, use_gaussian, all_in_gpu) -> Tuple[np.ndarray, np.ndarray]:
        
        verbose = True
        assert step_size <= 1
        assert self.get_device() != "cpu", "CPU not implemented"
        pad_border_mode = 'constant'
        pad_kwargs = {'constant_values': 0}

        if len(mirror_axes):
            if max(mirror_axes) > 2:
                raise ValueError("mirror axes. duh")

        if self.training:
            print('WARNING! Network is in train mode during inference. This may be intended, or not...')
        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"

        with autocast():
            with torch.no_grad():
                res = self._internal_predict_3D_3Dconv_tiled(x, step_size, do_mirroring, mirror_axes, patch_size,
                                                            use_gaussian, pad_border_mode, pad_kwargs, all_in_gpu, verbose)
        return res


    def _internal_predict_3D_3Dconv_tiled(self, x, step_size, do_mirroring, mirror_axes, patch_size, 
                                        use_gaussian, pad_border_mode, pad_kwargs, all_in_gpu, verbose) -> Tuple[np.ndarray, np.ndarray]:
        assert len(x.shape) == 4, "x must be (c, x, y, z)"
        assert self.get_device() != "cpu"

        torch.cuda.empty_cache()
        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape  # still c, x, y, z

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        # we only need to compute that once. It can take a while to compute this due to the large sigma in gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_3d is None or not all([i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_3d)]):
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)
                self._gaussian_3d = gaussian_importance_map
                self._patch_size_for_gaussian_3d = patch_size
            else:
                gaussian_importance_map = self._gaussian_3d
            gaussian_importance_map = torch.from_numpy(gaussian_importance_map).cuda(self.get_device(), non_blocking=True)
        else:
            gaussian_importance_map = None

        if all_in_gpu:
            if use_gaussian and num_tiles > 1:
                gaussian_importance_map = gaussian_importance_map.half()
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[gaussian_importance_map != 0].min()
                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(data.shape[1:], device=self.get_device())

            if verbose: print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half, device=self.get_device())

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half, device=self.get_device())
        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_3d
            else:
                add_for_nb_of_preds = np.ones(patch_size.shape[1:], dtype=np.float32)

            aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]
                    
                    predicted_patch = self._internal_maybe_mirror_and_pred_3D(data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], mirror_axes, do_mirroring, gaussian_importance_map)[0]

                    if all_in_gpu:
                        predicted_patch = predicted_patch.half()
                    else:
                        predicted_patch = predicted_patch.cpu().numpy()

                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                    aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds

        slicer = tuple([slice(0, aggregated_results.shape[i]) for i in range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        class_probabilities = aggregated_results / aggregated_nb_of_predictions
        
        predicted_segmentation = class_probabilities.argmax(0)

        if all_in_gpu:
            if verbose: print("copying results to CPU")
            predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
            class_probabilities = class_probabilities.detach().cpu().numpy()

        return predicted_segmentation, class_probabilities


    @staticmethod
    def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> List[List[int]]:
        assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
        assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

        target_step_sizes_in_voxels = [i * step_size for i in patch_size]
        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]
        
        steps = []
        for dim in range(len(patch_size)):
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999
            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

            steps.append(steps_here)
        return steps


    @staticmethod
    def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        from scipy.ndimage.filters import gaussian_filter
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(gaussian_importance_map[gaussian_importance_map != 0])

        return gaussian_importance_map


    def _internal_maybe_mirror_and_pred_3D(self, x, mirror_axes, do_mirroring, mult) -> torch.tensor:
        assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'
        x = to_cuda(maybe_to_torch(x), gpu_id=self.get_device())
        result_torch = torch.zeros([1, self.num_classes] + list(x.shape[2:]), dtype=torch.float).cuda(self.get_device(), non_blocking=True)

        if mult is not None:
            mult = to_cuda(maybe_to_torch(mult), gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1


        for m in range(mirror_idx):
            if m == 0:
                pred = softmax_helper(self(x)[0])
                result_torch += 1 / num_results * pred

            if m == 1 and (2 in mirror_axes):
                pred = softmax_helper(self(torch.flip(x, (4, )))[0])
                result_torch += 1 / num_results * torch.flip(pred, (4,))

            if m == 2 and (1 in mirror_axes):
                pred = softmax_helper(self(torch.flip(x, (3, )))[0])
                result_torch += 1 / num_results * torch.flip(pred, (3,))

            if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                pred = softmax_helper(self(torch.flip(x, (4, 3)))[0])
                result_torch += 1 / num_results * torch.flip(pred, (4, 3))

            if m == 4 and (0 in mirror_axes):
                pred = softmax_helper(self(torch.flip(x, (2, )))[0])
                result_torch += 1 / num_results * torch.flip(pred, (2,))

            if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                pred = softmax_helper(self(torch.flip(x, (4, 2)))[0])
                result_torch += 1 / num_results * torch.flip(pred, (4, 2))

            if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = softmax_helper(self(torch.flip(x, (3, 2)))[0])
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                pred = softmax_helper(self(torch.flip(x, (4, 3, 2)))[0])
                result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class ConvDropoutNormNonlin(nn.Module):
    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_op = conv_op
        self.conv_kwargs = conv_kwargs

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs['p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):

        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_op = conv_op
        self.conv_kwargs = conv_kwargs
        
        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


class TMV_UNet(SegmentationNetwork):
    def __init__(self, num_classes, act='leakyrelu', pool_op_kernel_sizes=None, conv_kernel_sizes=None, args=None): 
        super(TMV_UNet, self).__init__()
        self.args = args
        if self.args.ted:
            input_features = 2
        else:
            input_features = 1

        output_features = 32
        num_conv_per_stage = 2
        self.num_classes = num_classes

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        if self.pool_op_kernel_sizes is None:
            self.pool_op_kernel_sizes = [(2, 2, 2)] * 5
        if self.conv_kernel_sizes is None:
            self.conv_kernel_sizes = [(3, 3, 3)] * 6

        self.max_num_features = 320

        self.conv_op = nn.Conv3d
        self.nonlin = nn.LeakyReLU
        
        if act == 'relu':
            self.nonlin_kwargs = {'negative_slope': 0, 'inplace': True}
        elif act == 'leakyrelu':
            self.nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        else:
            print('***** please check nonlin act !!! *****')

        self.dropout_op = nn.Dropout3d
        self.dropout_op_kwargs = {'p': 0, 'inplace': True}
        self.norm_op = nn.InstanceNorm3d
        self.norm_op_kwargs = {'eps': 1e-5, 'affine': True}

        self.input_shape_must_be_divisible_by = np.prod(self.pool_op_kernel_sizes, 0, dtype=np.int64)
        
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.tu = []
        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        # encoder
        for d in range(len(self.pool_op_kernel_sizes)):
            if d != 0:
                first_stride = self.pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]

            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=ConvDropoutNormNonlin))
            input_features = output_features
            output_features = int(np.round(output_features * 2))
            output_features = min(output_features, self.max_num_features)

        # bottleneck
        first_stride = self.pool_op_kernel_sizes[-1]
        final_num_features = output_features
        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[len(self.pool_op_kernel_sizes)]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[len(self.pool_op_kernel_sizes)]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=ConvDropoutNormNonlin),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=ConvDropoutNormNonlin)))

        # decoder1, mask
        for u in range(len(self.pool_op_kernel_sizes)):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].output_channels  
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            final_num_features = nfeatures_from_skip
            self.tu.append(nn.ConvTranspose3d(nfeatures_from_down, nfeatures_from_skip, self.pool_op_kernel_sizes[-(u + 1)],
                                          self.pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=ConvDropoutNormNonlin),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=ConvDropoutNormNonlin)
            ))
        self.seg_outputs_1 = nn.Conv3d(self.conv_blocks_localization[len(self.conv_blocks_localization)-1][-1].output_channels, num_classes, 1, 1, 0, 1, 1, False)
        self.seg_outputs_2 = nn.Conv3d(self.conv_blocks_localization[len(self.conv_blocks_localization)-1][-1].output_channels, 1, 1, 1, 0, 1, 1, False)

        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.tu = nn.ModuleList(self.tu)
    
        self.weightInitializer = InitWeights_He(1e-2)
        self.apply(self.weightInitializer)

    def forward(self, x, ret_intermediate=False):
        skips = []
        encoder_attentions = []
        bottleneck_attentions = []
        decoder_attentions = []
        # encoder
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if ret_intermediate:
                encoder_attentions.append(x)

        # bottleneck
        x = self.conv_blocks_context[-1](x)
        if ret_intermediate:
            bottleneck_attentions.append(x)


        # decoder
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            if ret_intermediate:
                decoder_attentions.append(x)

        seg_output1 = self.seg_outputs_1(x)
        seg_output2 = self.seg_outputs_2(x)


        if ret_intermediate == False:
            return seg_output1, seg_output2
        else:
            attentions = {'encoder': encoder_attentions, 'bottleneck': bottleneck_attentions, 'decoder': decoder_attentions}
            return seg_output1, seg_output2, attentions




def pad_nd_image(image, new_shape=None, mode="constant", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):

    if kwargs is None:
        kwargs = {'constant_values': 0}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        res = np.pad(image, pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer


if __name__ == '__main__':
    pass



