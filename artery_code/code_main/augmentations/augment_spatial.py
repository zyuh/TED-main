import numpy as np
from scipy.ndimage import map_coordinates
import torch
import abc


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """
    def __call__(self, sample):
        image, label = sample['data'], sample['seg']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'data': image, 'seg': label}


def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))

    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)

    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)

    if len(coords) == 3: # 3d
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
    elif len(coords) == 2: # 伪2d
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)

    final_shape /= min(scale_range)
    return final_shape.astype(int)


def augment_spatial(data, seg, patch_size,
                    do_rotation=True, 
                    angle_x=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi), 
                    angle_y=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    angle_z=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    do_scale=True, scale=(0.7, 1.4), do_dummy_2D_aug=False):
    if do_dummy_2D_aug:
        dim = 2
        patch_size = patch_size[1:]
        angle_x = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
        seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        dim = 3
        seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]), dtype=np.float32)
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]), dtype=np.float32)

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = False

        # 随机旋转
        if do_rotation and np.random.uniform() < 0.2:
            a_x = np.random.uniform(angle_x[0], angle_x[1])
            a_y = np.random.uniform(angle_y[0], angle_y[1])
            a_z = np.random.uniform(angle_z[0], angle_z[1])
            if do_dummy_2D_aug:
                coords = rotate_coords_2d(coords, a_x)
            else:
                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
            modified_coords = True

        # 随机缩放
        if do_scale and np.random.uniform() < 0.2:
            if np.random.random() < 0.5:
                sc = np.random.uniform(scale[0], 1)
            else:
                sc = np.random.uniform(1, scale[1])

            coords = scale_coords(coords, sc)
            modified_coords = True

        if modified_coords:
            for d in range(dim):
                ctr = int(np.round(data.shape[d + 2] / 2.)) #center
                coords[d] += ctr
            for channel_id in range(data.shape[1]): #沿channel
                data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, 3, 'nearest', cval=0)
            for channel_id in range(seg.shape[1]):
                seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, 0, 'constant', cval=0, is_seg=True)
        else:
            s = seg[sample_id:sample_id + 1]
            d, s = center_crop_aug(data[sample_id:sample_id + 1], patch_size, s)
            data_result[sample_id] = d[0]
            seg_result[sample_id] = s[0]

    return data_result, seg_result


def augment_spatialv2(data, seg, density, patch_size,
                    do_rotation=True, 
                    angle_x=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi), 
                    angle_y=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    angle_z=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    do_scale=True, scale=(0.7, 1.4), do_dummy_2D_aug=False):
    # data bcxyz
    if do_dummy_2D_aug:
        dim = 2
        patch_size = patch_size[1:]
        angle_x = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
        seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        dim = 3
        seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]), dtype=np.float32)
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]), dtype=np.float32)
        density_result = np.zeros((density.shape[0], density.shape[1], patch_size[0], patch_size[1], patch_size[2]), dtype=np.float32)

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = False

        # 随机旋转
        if do_rotation and np.random.uniform() < 0.2:
            a_x = np.random.uniform(angle_x[0], angle_x[1])
            a_y = np.random.uniform(angle_y[0], angle_y[1])
            a_z = np.random.uniform(angle_z[0], angle_z[1])
            if do_dummy_2D_aug:
                coords = rotate_coords_2d(coords, a_x)
            else:
                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
            modified_coords = True

        # 随机缩放
        if do_scale and np.random.uniform() < 0.2:
            if np.random.random() < 0.5:
                sc = np.random.uniform(scale[0], 1)
            else:
                sc = np.random.uniform(1, scale[1])

            coords = scale_coords(coords, sc)
            modified_coords = True

        if modified_coords:
            for d in range(dim):
                ctr = int(np.round(data.shape[d + 2] / 2.)) #center
                coords[d] += ctr
            for channel_id in range(data.shape[1]): #沿channel
                data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, 3, 'nearest', cval=0)
            for channel_id in range(seg.shape[1]):
                seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, 0, 'constant', cval=0, is_seg=True)
                # density 的 插值与 data 一样比较好, 但channel数是和seg相同的
                density_result[sample_id, channel_id] = interpolate_img(density[sample_id, channel_id], coords, 3, 'nearest', cval=0)              
        else:
            s = seg[sample_id:sample_id + 1]
            den = density[sample_id:sample_id + 1]
            # d, s = center_crop_aug(data[sample_id:sample_id + 1], patch_size, s)
            d, s, den = center_crop_augv2(data[sample_id:sample_id + 1], patch_size, s, den)
            data_result[sample_id] = d[0]
            seg_result[sample_id] = s[0]
            density_result[sample_id] = den[0]

    return data_result, seg_result, density_result


def create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
    return coords


def rotate_coords_3d(coords, angle_x, angle_y, angle_z):
    rot_matrix = np.identity(len(coords))
    rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
    rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
    rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)
    coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
    return coords


def rotate_coords_2d(coords, angle):
    rot_matrix = create_matrix_rotation_2d(angle)
    coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
    return coords


def create_matrix_rotation_2d(angle, matrix=None):
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    if matrix is None:
        return rotation
    return np.dot(matrix, rotation)


def create_matrix_rotation_x_3d(angle, matrix=None):
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])
    if matrix is None:
        return rotation_x
    return np.dot(matrix, rotation_x)


def create_matrix_rotation_y_3d(angle, matrix=None):
    rotation_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                           [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]])
    if matrix is None:
        return rotation_y
    return np.dot(matrix, rotation_y)


def create_matrix_rotation_z_3d(angle, matrix=None):
    rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]])
    if matrix is None:
        return rotation_z
    return np.dot(matrix, rotation_z)


def scale_coords(coords, scale):
    if isinstance(scale, (tuple, list, np.ndarray)):
        assert len(scale) == len(coords)
        for i in range(len(scale)):
            coords[i] *= scale[i]
    else:
        coords *= scale
    return coords


def interpolate_img(img, coords, order=3, mode='nearest', cval=0.0, is_seg=False):
    if is_seg and order != 0:
        unique_labels = np.unique(img)
        result = np.zeros(coords.shape[1:], img.dtype)
        for i, c in enumerate(unique_labels):
            res_new = map_coordinates((img == c).astype(float), coords, order=order, mode=mode, cval=cval)
            result[res_new >= 0.5] = c
        return result
    else:
        return map_coordinates(img.astype(float), coords, order=order, mode=mode, cval=cval).astype(img.dtype)


def center_crop_aug(data, crop_size, seg=None):
    return crop(data, seg, crop_size, 'center')

def center_crop_augv2(data, crop_size, seg=None, density=None):
    return cropv2(data, seg, density, crop_size, 'center')


def crop(data, seg=None, crop_size=128, crop_type="center",
         pad_mode='constant', pad_kwargs={'constant_values': 0},
         pad_mode_seg='constant', pad_kwargs_seg={'constant_values': 0}):
    # crop_size = patch size
    data_shape = tuple([len(data)] + list(data[0].shape))
    seg_shape = tuple([len(seg)] + list(seg[0].shape))
    dim = len(data_shape) - 2
    data_return = np.zeros([data_shape[0], data_shape[1]] + list(crop_size), dtype=data[0].dtype)
    seg_return = np.zeros([seg_shape[0], seg_shape[1]] + list(crop_size), dtype=seg[0].dtype)

    for b in range(data_shape[0]):
        data_shape_here = [data_shape[0]] + list(data[b].shape)
        seg_shape_here = [seg_shape[0]] + list(seg[b].shape)
        lbs = get_lbs_for_center_crop(crop_size, data_shape_here)
        need_to_pad = [[0, 0]] + [[abs(min(0, lbs[d])),
                                   abs(min(0, data_shape_here[d + 2] - (lbs[d] + crop_size[d])))]
                                  for d in range(dim)]

        # we should crop first, then pad -> reduces i/o for memmaps, reduces RAM usage and improves speed
        ubs = [min(lbs[d] + crop_size[d], data_shape_here[d+2]) for d in range(dim)]
        lbs = [max(0, lbs[d]) for d in range(dim)]

        slicer_data = [slice(0, data_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
        data_cropped = data[b][tuple(slicer_data)]
        slicer_seg = [slice(0, seg_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
        seg_cropped = seg[b][tuple(slicer_seg)]

        if any([i > 0 for j in need_to_pad for i in j]):
            data_return[b] = np.pad(data_cropped, need_to_pad, pad_mode, **pad_kwargs)
            if seg_return is not None:
                seg_return[b] = np.pad(seg_cropped, need_to_pad, pad_mode_seg, **pad_kwargs_seg)
        else:
            data_return[b] = data_cropped
            if seg_return is not None:
                seg_return[b] = seg_cropped

    return data_return, seg_return


def cropv2(data, seg=None, density=None, crop_size=128, crop_type="center",
         pad_mode='constant', pad_kwargs={'constant_values': 0},
         pad_mode_seg='constant', pad_kwargs_seg={'constant_values': 0}):
    # crop_size = patch size
    data_shape = tuple([len(data)] + list(data[0].shape))
    seg_shape = tuple([len(seg)] + list(seg[0].shape))
    density_shape = tuple([len(density)] + list(density[0].shape))
    dim = len(data_shape) - 2
    # assert dim == 3
    data_return = np.zeros([data_shape[0], data_shape[1]] + list(crop_size), dtype=data[0].dtype)
    seg_return = np.zeros([seg_shape[0], seg_shape[1]] + list(crop_size), dtype=seg[0].dtype)
    density_return = np.zeros([density_shape[0], density_shape[1]] + list(crop_size), dtype=density[0].dtype)

    for b in range(data_shape[0]):
        data_shape_here = [data_shape[0]] + list(data[b].shape)
        seg_shape_here = [seg_shape[0]] + list(seg[b].shape)
        density_shape_here = [density_shape[0]] + list(density[b].shape)

        lbs = get_lbs_for_center_crop(crop_size, data_shape_here)
        need_to_pad = [[0, 0]] + [[abs(min(0, lbs[d])),
                                   abs(min(0, data_shape_here[d + 2] - (lbs[d] + crop_size[d])))]
                                  for d in range(dim)]

        # we should crop first, then pad -> reduces i/o for memmaps, reduces RAM usage and improves speed
        ubs = [min(lbs[d] + crop_size[d], data_shape_here[d+2]) for d in range(dim)]
        lbs = [max(0, lbs[d]) for d in range(dim)]

        slicer_data = [slice(0, data_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
        data_cropped = data[b][tuple(slicer_data)]
        slicer_seg = [slice(0, seg_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
        seg_cropped = seg[b][tuple(slicer_seg)]

        slicer_density = [slice(0, density_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
        density_cropped = density[b][tuple(slicer_density)]

        if any([i > 0 for j in need_to_pad for i in j]):
            data_return[b] = np.pad(data_cropped, need_to_pad, pad_mode, **pad_kwargs)
            if seg_return is not None:
                seg_return[b] = np.pad(seg_cropped, need_to_pad, pad_mode_seg, **pad_kwargs_seg)
            if density_return is not None:
                density_return[b] = np.pad(density_cropped, need_to_pad, pad_mode_seg, **pad_kwargs_seg)
        else:
            data_return[b] = data_cropped
            if seg_return is not None:
                seg_return[b] = seg_cropped
            if density_return is not None:
                density_return[b] = density_cropped

    return data_return, seg_return, density_return


def get_lbs_for_center_crop(crop_size, data_shape):
    lbs = []
    for i in range(len(data_shape) - 2):
        lbs.append((data_shape[i + 2] - crop_size[i]) // 2)
    return lbs

