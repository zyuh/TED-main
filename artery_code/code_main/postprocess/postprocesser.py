import ast
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label

def remove_all_but_the_largest_connected_component(image):

    mask = np.zeros_like(image, dtype=bool)
    mask[image == 1] = True

    lmap, num_objects = label(mask.astype(int))  #num_objects表示连通域的个数；  lmap=按连通域将每块连通域标注为1，2。。。。

    object_sizes = {}
    for object_id in range(1, num_objects + 1):
        object_sizes[object_id] = (lmap == object_id).sum()  #GetSpacing

    if num_objects > 0:
        size_list = list(object_sizes.values())
        size_list.sort()
        maximum_size = max(size_list)
        assert maximum_size == size_list[-1]

        for object_id in range(1, num_objects + 1):
            if object_sizes[object_id] != maximum_size:
                remove = True
                if remove:
                    image[(lmap == object_id) & mask] = 0  #满足remove条件的连通域置为0，保存下来的是Image

    return image
