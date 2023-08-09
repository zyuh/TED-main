import SimpleITK as sitk 
import os
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import torch
from multiprocessing import Pool
import time
import warnings
warnings.filterwarnings("ignore")

# tools
def maybe_mkdir_p(directory):
    directory = os.path.abspath(directory)
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i+1])):
            try:
                os.mkdir(os.path.join("/", *splits[:i+1]))
            except FileExistsError:
                # this can sometimes happen when two jobs try to create the same directory at the same time,
                # especially on network drives.
                print("WARNING: Folder %s already existed and does not need to be created" % directory)


# tools for ''_complete_flatten''
def _recursive_search_window(idx, post_flatten_tmp, init_len_window):
    half = int((init_len_window - 1) / 2)
    if idx < half:
        window = np.array(
            (post_flatten_tmp[idx - half:]).tolist() + (post_flatten_tmp[0:idx + half + 1]).tolist())
        # print('window', window)
    elif idx >= len(post_flatten_tmp) - half:
        window = np.array((post_flatten_tmp[idx - half:]).tolist() + (
            post_flatten_tmp[0:idx - len(post_flatten_tmp) + half + 1]).tolist())
    else:
        window = post_flatten_tmp[idx - half:idx + half + 1]
    if np.any(window > 0):  # and not (window[0] == 0 or window[-1]==0):
        # it is better to hit the new_id when both two end of the window is not zero., but results show vice verse
        counts = np.bincount(window)
        new_id = np.argmax(counts[1:]) + 1
        post_flatten_tmp[idx] = int(new_id)
        # print("satisfying", post_flatten_smooth)
    else:
        post_flatten_tmp = _recursive_search_window(idx, post_flatten_tmp, init_len_window + 2)
    return post_flatten_tmp


def _recursive_search(idx, theta1_int, r1, theta1, len_window):
    # default len_window = 9
    neighbor = np.zeros((len_window, 2))
    half = int((len_window - 1) / 2)  # 4
    # print("start window..", len_window)
    for ii in range(len_window):
        idx_local = idx - half + ii
        if idx_local < 0:
            idx_local = 360 + idx_local
        elif idx_local > 360:
            idx_local = idx_local - 360
        tmp_r = np.array([r1[i] for i in np.where(theta1_int == np.float32(idx_local))])
        tmp_t = np.array([theta1[i] for i in np.where(theta1_int == np.float32(idx_local))])

        sortind = np.argsort(tmp_r)[0]
        tmp_r = np.array([tmp_r[0][id] for id in sortind])
        tmp_t = np.array([tmp_t[0][id] for id in sortind])  # almost the same number, not round

        tmp_r_len = len(tmp_r)
        # print('len(tmp_r)', len(tmp_r))
        if tmp_r_len != 0:
            tmp_r_mid, tmp_t_mid = tmp_r[int(len(tmp_r) / 2)], [int(len(tmp_t) / 2)]
            neighbor[ii, 0] = tmp_r_len
            neighbor[ii, 1] = tmp_r_mid
            # neighbor[ii, 1] = tmp_t_mid
        else:
            neighbor[ii, 0] = np.nan
            neighbor[ii, 1] = np.nan
    if np.all(np.isnan(neighbor)):
        # scale up the search scope! recursive searching?
        avg_window_width, avg_window_mid = _recursive_search(idx, theta1_int, r1, theta1, len_window=len_window + 2)
    else:
        avg_window_width = np.nanmean(neighbor[:, 0])
        avg_window_mid = np.nanmean(neighbor[:, 1])
    return avg_window_width, avg_window_mid


# 1、极坐标转换
def cart2polar(mask, M, cap, recover_flag=False, cap2=None, prgt_flag=False, args=None):
    # prgt_flag = True 表示使用预测的mask进行转换，否则使用gt mask进行转换
    if cap2 is not None:
        prgt_flag = True
        flatten_prgt = np.zeros((360, args.ring_width))
    cart_x, cart_y = np.nonzero(mask)
    cart_x, cart_y = np.array(cart_x).astype(np.float32), np.array(cart_y).astype(np.float32)
    # 计算重心
    cY, cX = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    cart_offset_x, cart_offset_y = cart_x - cX, cart_y - cY
    r1, theta1 = cv2.cartToPolar(cart_offset_x, cart_offset_y, angleInDegrees=True)
    r1, theta1 = np.array(r1)[:, 0], np.array(theta1)[:, 0]
    theta1_int = [np.round(x) for x in theta1]
    count_len = []
    flatten = np.zeros((360, args.ring_width))
    for idx in np.arange(0, 360):
        tmp_r = np.array([r1[i] for i in np.where(theta1_int == np.float32(idx))])
        tmp_t = np.array([theta1[i] for i in np.where(theta1_int == np.float32(idx))])
        sortind = np.argsort(tmp_r)[0]
        tmp_r = np.array([tmp_r[0][id] for id in sortind])
        tmp_t = np.array([tmp_t[0][id] for id in sortind])
        if len(tmp_r) == 0: 
            continue
        count_len.append(len(tmp_r))
        x_new, y_new = cv2.polarToCart(tmp_r, tmp_t, angleInDegrees=True)
        x_cart_new, y_cart_new = x_new + cX, y_new + cY
        x_cart_new = [np.uint(np.round(x)) for x in x_cart_new]
        y_cart_new = [np.uint(np.round(y)) for y in y_cart_new]
        tmp_list = np.zeros(args.n_classes - 1)

        for i in range(len(y_cart_new)):
            flatten[idx][i] = cap[np.int(x_cart_new[i])][np.int(y_cart_new[i])]
            if prgt_flag:
                flatten_prgt[idx][i] = cap2[np.int(x_cart_new[i])][np.int(y_cart_new[i])]

    if prgt_flag and recover_flag:
        return flatten, flatten_prgt, r1, theta1
    elif not prgt_flag and recover_flag:
        return flatten, r1, theta1
    elif not prgt_flag and not recover_flag:
        return flatten
    else:
        assert 1 == 2, 'error return'


# 2、消歧
def post(tmp_stack, args=None):
    new_tmp_stack = []
    for i in range(len(tmp_stack)):
        counts = np.bincount(tmp_stack[i].astype(np.int))
        if len(counts) == 1:
            new_tmp_stack.append(0)
            continue
        new_tmp_stack.append(np.argmax(counts[1:]) + 1)
    new_tmp_stack = np.array(new_tmp_stack)
    new_tmp_stack = np.repeat(np.reshape(new_tmp_stack, (new_tmp_stack.shape[0], 1)), args.ring_width,
                              axis=1)
    return new_tmp_stack

# 3、平滑
def _smooth_flatten(post_flatten, args=None):
    post_flatten_smooth = np.zeros((len(post_flatten), args.n_classes))
    for i in range(len(post_flatten)):
        for j in range(args.n_classes):
            post_flatten_smooth[i][j] = 1 if post_flatten[i][0] == j else 0
    post_flatten_smooth = torch.from_numpy(post_flatten_smooth).permute(1, 0).unsqueeze(0)
    post_flatten_smooth = torch.nn.functional.interpolate(post_flatten_smooth, scale_factor=1 / 8,
                                                    mode='linear')
    post_flatten_smooth = torch.nn.functional.interpolate(post_flatten_smooth, scale_factor=8,
                                                    mode='nearest').permute(2, 1, 0).repeat(1, 1,
                                                                                            args.ring_width)
    post_flatten_smooth = torch.argmax(post_flatten_smooth, dim=1).numpy()
    return post_flatten_smooth


# 4、补完
def _complete_flatten(post_flatten_tmp, r1, theta1,args=None):
    post_flatten_tmp = post_flatten_tmp[:, 0]
    theta1_int = [np.round(x) for x in theta1]
    for idxx in range(360):
        tmp = np.array([r1[i] for i in np.where(theta1_int == np.float32(idxx))])
        if len(tmp[0]) != 0 and post_flatten_tmp[idxx] != 0:
            continue
        avg_window_width, avg_window_mid = _recursive_search(idxx, theta1_int, r1, theta1,
                                                             len_window=49)
        if post_flatten_tmp[idxx] == 0:
            post_flatten_tmp = _recursive_search_window(idxx, post_flatten_tmp, init_len_window=9)
        r1 = np.concatenate((r1, np.arange(avg_window_mid - int(avg_window_width / 2),
                                           avg_window_mid + int(avg_window_width / 2) + 1)))
        len_new = 2 * int(avg_window_width / 2) + 1
        assert len(np.repeat(np.array([idxx]), len_new,
                             axis=0)) == len_new, "the new r1 and new theta1 have not same length"
        theta1 = np.concatenate((theta1, np.repeat(np.array([idxx]).astype(np.float), len_new, axis=0)))
    post_flatten_complete = np.repeat(np.reshape(post_flatten_tmp, (post_flatten_tmp.shape[0], 1)),
                                      args.ring_width, axis=1)
    return post_flatten_complete, r1, theta1

# 5、还原回直角坐标系
def degree2cart(idx, name_b, cvt_r1, cvt_theta1, theta1_int, cX_cvt_gt, cY_cvt_gt):
    tmp_r = np.array([cvt_r1[i] for i in np.where(theta1_int == np.float32(idx))])
    tmp_t = np.array([cvt_theta1[i] for i in np.where(theta1_int == np.float32(idx))])
    sortind = np.argsort(tmp_r)[0]
    tmp_r = [tmp_r[0][id] for id in sortind]
    tmp_t = [tmp_t[0][id] for id in sortind]  # the same numbers
    if len(tmp_r) == 0: 
        print("the boundary is not complete, name {} degree {} ".format(name_b, idx % 4))
        assert 1 == 2
    tmp_r, tmp_t = np.array(tmp_r), np.array(tmp_t)
    x_new, y_new = cv2.polarToCart(tmp_r, tmp_t, angleInDegrees=True)
    x_cart_new, y_cart_new = x_new + cX_cvt_gt, y_new + cY_cvt_gt
    x_cart_new = [np.uint(np.round(x)) for x in x_cart_new]
    y_cart_new = [np.uint(np.round(y)) for y in y_cart_new]
    i_mid = int(len(x_cart_new) / 2)
    return x_cart_new, y_cart_new


# 可视化 【注意这个配色和 itk-snap 默认的不一样】
def vis(array, save_name, vis_center=False, center=None, center_gt=None):
    array_vis = np.zeros((array.shape[0], array.shape[1], 3))
    # opencv: bgr
    '''绿色，蓝色，红色分别对应 123'''
    array_vis[array == 1, 1] = 255 # green
    array_vis[array == 2, 0] = 255 # blue
    array_vis[array == 3, 2] = 255 # red
    if vis_center:
        cX, cY = center
        array_vis[cX:cX + 5, cY:cY + 5, 1] = 255
        if center_gt is not None:
            cX_gt, cY_gt = center
            array_vis[cX_gt:cX_gt + 5, cY_gt:cY_gt + 5, 2] = 255  # red
    cv2.imwrite((save_name+'.png'), array_vis)

def show_npz(npz_path, save_path):
    npz_file = np.load(npz_path)
    name = npz_path.split('/')[-1][:-4]

    img_slice = npz_file['image']
    roi_slice = npz_file['roi']
    roi_degree = npz_file['roi_degree']

    maybe_mkdir_p(os.path.join(save_path, name))
    cv2.imwrite(os.path.join(save_path, name, 'img_slice.png'), img_slice)
    cv2.imwrite(os.path.join(save_path, name, 'roi_slice.png'), (roi_slice/np.max(roi_slice)*255).astype(np.uint8))
    cv2.imwrite(os.path.join(save_path, name, 'roi_degree.png'), (fen_degree/np.max(fen_degree)*255).astype(np.uint8))

# 单个样本的单个slice的cap/fen的转换,以cap命名
def label_convert_for_each_slice(roi_slice, name, args):
    label_hcc_b, name_b = roi_slice, name
    mask_gt = (label_hcc_b > 0).astype(np.float)
    label_recover_degree_b = np.zeros_like(label_hcc_b)

    if np.sum(mask_gt)==0: 
        raise ValueError
    # 求质心
    M_gt = cv2.moments(mask_gt)

    # 根据roi得到边缘 
    dilation = cv2.dilate(label_hcc_b.astype(np.uint8), kernel=np.ones((3, 3), np.uint8), iterations=1)
    erosion = cv2.erode(label_hcc_b.astype(np.uint8), kernel=np.ones((3, 3), np.uint8), iterations=1)
    hcc_cvt_edge_b = (dilation - erosion).astype(np.float)
    hcc_cvt_edge_b = (cv2.GaussianBlur(hcc_cvt_edge_b, (5, 5), 0) > 0).astype(np.float)

    M_cvt_gt = cv2.moments(label_hcc_b)  # to make online recover easier! replace hcc_cvt_edge_b
    if np.sum(hcc_cvt_edge_b) == 0:  # data to be clean
        assert 1 == 2, "np.sum(hcc_cvt_edge_b) == 0, name {}".format(name_b)

    cY_cvt_gt, cX_cvt_gt = int(M_cvt_gt["m10"] / M_cvt_gt["m00"]), int(M_cvt_gt["m01"] / M_cvt_gt["m00"])
    # 极坐标转换
    flatten_cvt_gt, cvt_r1, cvt_theta1 = cart2polar(hcc_cvt_edge_b, M_cvt_gt, hcc_cvt_edge_b, recover_flag=True, args=args)
    theta1_int = [np.round(x) for x in cvt_theta1] #取整
    # complete!
    check_complete = np.array([len(np.array([cvt_r1[i] for i in np.where(theta1_int == np.float32(idx))])[0]) == 0 for idx in range(360)])
    if np.any(check_complete):
        post_flatten_cvt_gt = post(flatten_cvt_gt, args=args)
        post_flatten_cvt_gt = _smooth_flatten(post_flatten_cvt_gt, args=args)
        post_flatten_cvt_gt, cvt_r1, cvt_theta1 = _complete_flatten(post_flatten_cvt_gt.copy(), cvt_r1, cvt_theta1, args=args)
        theta1_int = [np.round(x) for x in cvt_theta1]
    
    # 转换回直角坐标
    for idx_scale4 in range(90):
        idx = idx_scale4 * 4
        idx_prev = idx - 1 if idx > 0 else 359
        idx_next = idx + 1

        x_cart_new, y_cart_new = degree2cart(idx, name_b, cvt_r1, cvt_theta1, theta1_int, cX_cvt_gt, cY_cvt_gt)
        x_cart_new_prev, y_cart_new_prev = degree2cart(idx_prev, name_b, cvt_r1, cvt_theta1, theta1_int, cX_cvt_gt, cY_cvt_gt)
        x_cart_new_next, y_cart_new_next = degree2cart(idx_next, name_b, cvt_r1, cvt_theta1, theta1_int, cX_cvt_gt, cY_cvt_gt)

        for i in range(len(y_cart_new)):
            label_recover_degree_b[np.int(x_cart_new[i])][np.int(y_cart_new[i])] = idx_scale4 + 1

        for i in range(len(y_cart_new_prev)):
            label_recover_degree_b[np.int(x_cart_new_prev[i])][np.int(y_cart_new_prev[i])] = idx_scale4 + 1

        for i in range(len(y_cart_new_next)):
            label_recover_degree_b[np.int(x_cart_new_next[i])][np.int(y_cart_new_next[i])] = idx_scale4 + 1 

    return label_recover_degree_b


# 单个样本的转换
def main_label_convert(img_path, roi_path, each_uid_sample, save_path, args):
    img_itk = sitk.ReadImage(img_path)
    img_npy = sitk.GetArrayFromImage(img_itk)
    roi_itk = sitk.ReadImage(roi_path)
    roi_npy = sitk.GetArrayFromImage(roi_itk)
    # print(img_npy.shape, roi_npy.shape)

    for slice_th in range(len(img_npy)):
        img_slice = img_npy[slice_th]
        roi_slice = roi_npy[slice_th]
        # print(roi_path, slice_th, np.sum(roi_slice))
        if np.sum(roi_slice) < 3500:
            continue
        name = each_uid_sample+'_slice_'+str(slice_th)
        roi_degree = label_convert_for_each_slice(roi_slice, name, args)

        save_path_final = os.path.join(save_path, each_uid_sample, name) # slice是从0开始
        maybe_mkdir_p(os.path.join(save_path, each_uid_sample))
        np.savez(save_path_final+'.npz', image=img_slice, roi=roi_slice, roi_degree=roi_degree)


# 多进程快速处理
def get_all_capfen_5mm_slice_multi_pool(args):

    raw_data_root_path = args.data_path
    uid_sample_list = os.listdir(raw_data_root_path)
    uid_sample_list.sort()

    samples_list = []
    for each_uid_sample in uid_sample_list:
        maybe_mkdir_p(os.path.join(raw_data_root_path, each_uid_sample))
        maybe_mkdir_p(os.path.join(args.save_path, each_uid_sample))

        img_path = os.path.join(raw_data_root_path, each_uid_sample, 'vein_img.nii')
        # roi_path = os.path.join(args.save_path, each_uid_sample, each_uid_sample+'_roi.nii')# gt
        roi_path = os.path.join(args.save_path, each_uid_sample, each_uid_sample+'_vein_roi_vein.nii.gz')

        # 标注是否齐全
        if not os.path.exists(img_path):
            continue
        # if not os.path.exists(roi_path):
        #     continue

        each_sample = [img_path, roi_path, each_uid_sample, args.save_path, args]
        samples_list.append(each_sample)
    
    not_exists_flag = True
    while not_exists_flag:
        check_list = [0] * len(samples_list)
        for each_sample_th in range(len(samples_list)):
            each_sample = samples_list[each_sample_th]
            each_roi_path = each_sample[1]
            if os.path.exists(each_roi_path):
                check_list[each_sample_th] = 1
        if np.all(check_list):
            not_exists_flag = False
        time.sleep(5)

    p = Pool(args.num_thread)
    return_data = p.starmap(main_label_convert, samples_list)
    p.close()
    p.join()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ring_width', type=int, default=150, help='max ring width')
    parser.add_argument('--n_classes', type=int, default=4, help='n_classes')
    parser.add_argument('--data_path', type=str, default='', help='raw_data_path')
    parser.add_argument('--save_path', type=str, default='', help='where to save')
    parser.add_argument('--num_thread', type=int, default=1, help='multi-process')
    parser.add_argument('--slice_path', type=str, default='', help='vis which slice')
    parser.add_argument('--vis_save_path', type=str, default='', help='where to save')
    parser.add_argument('--show', action='store_true', help='vis')
    args = parser.parse_args()
    
    if args.show:
        # 可视化，看效果
        show_npz(args.slice_path, args.vis_save_path)
    else:
        get_all_capfen_5mm_slice_multi_pool(args)
        print('convert2npz finish!')
