import numpy as np
import cv2
import os
from utils.file_ops import *
from multiprocessing import Pool
import SimpleITK as sitk

def get_density_map_gaussian(im):
    # im为artery的binary标注
    im_density = np.zeros_like(im)
    [h,w] = im_density.shape

    points = list(zip(*np.nonzero(im)))

    if (len(points)==0):
        return im
    
    if (len(points)==1):
        return im

    for j in range(len(points)):
        f_sz = 15
        sigma = 4.0

        H = np.multiply(cv2.getGaussianKernel(f_sz, sigma), (cv2.getGaussianKernel(f_sz, sigma)).T) 
        x = points[j][0]
        y = points[j][1]

        x1 = x - f_sz//2
        y1 = y - f_sz//2
        x2 = x + f_sz//2 + 1
        y2 = y + f_sz//2 + 1

        assert x1 >= 0, print(x1)
        assert y1 >= 0, print(y1)
        assert x2 <= h, print(x2)
        assert y2 <= w, print(y2)

        im_density = im_density.astype(float)
        im_density[x1:x2, y1:y2] = im_density[x1:x2, y1:y2] +  H
    return im_density


def get_density_map_gaussian3d(im, f_sz=3, sigma=0.45):
    im = im.squeeze()
    im_density = np.zeros_like(im)
    [h,w,zz] = im_density.shape
    points = list(zip(*np.nonzero(im)))

    if (len(points)==0):
        return im[None]
    if (len(points)==1):
        return im[None]
    for j in range(len(points)):
        # 0.4-0.77 |  0.45-0.62
        kernel_1d = cv2.getGaussianKernel(f_sz, sigma)
        kernel_2d = np.multiply(cv2.getGaussianKernel(f_sz, sigma), (cv2.getGaussianKernel(f_sz, sigma)).T) 
        kernel_3d = np.multiply(kernel_1d[:,:,None], kernel_2d[None])
        H = kernel_3d
        # if j == 0:
        #     print(np.max(kernel_3d))
        x = points[j][0]
        y = points[j][1]
        z = points[j][2]

        x1 = x - f_sz//2
        x2 = x + f_sz//2 + 1
        y1 = y - f_sz//2
        y2 = y + f_sz//2 + 1
        z1 = z - f_sz//2
        z2 = z + f_sz//2 + 1

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        z1 = max(z1, 0)
        x2 = min(x2, h)
        y2 = min(y2, w)
        z2 = min(z2, zz)

        im_density = im_density.astype(float)

        if im_density[x1:x2, y1:y2, z1:z2].shape != H.shape:
            im_shape = im_density[x1:x2, y1:y2, z1:z2].shape
            H = H[:im_shape[0], :im_shape[1], :im_shape[2]]
        im_density[x1:x2, y1:y2, z1:z2] = im_density[x1:x2, y1:y2, z1:z2] +  H

    return im_density[None]



def get_density_and_save(each_artery_label, density_save_path, f_sz, sigma):
    print('making... ', each_artery_label)

    # img_lt = sitk.ReadImage(each_artery_label)
    # each_artery = sitk.GetArrayFromImage(img_lt) # 01
    each_artery = np.load(each_artery_label, 'r')

    each_artery_density = get_density_map_gaussian3d(each_artery, f_sz, sigma)
    # 不会改变数量
    # assert np.abs(np.sum(each_artery_density)-np.sum(each_artery)) < 1e-4, print(np.sum(each_artery_density), np.sum(each_artery))

    each_artery_label_id = each_artery_label.split('/')[-2]
    new_name_to_save = join(density_save_path, each_artery_label_id + '.npy')

    # print(f_sz, sigma, 'num change:', np.abs(np.sum(each_artery_density)-np.sum(each_artery)))
    print('saving... ', new_name_to_save)
    np.save(new_name_to_save, each_artery_density)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--f_sz', type=int, default=7)
    parser.add_argument('--sigma', type=int, default=2)
    parser.add_argument('--num_thread', type=int, default=8)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis_data_path', type=str, default=None)
    parser.add_argument('--vis_save_path', type=str, default=None)

    args = parser.parse_args()

    if not args.vis:
        maybe_mkdir_p(args.save_path)

        # 跳过已生成过的
        # maden_sample_list = subfiles(args.save_path, join=False, suffix='.npy')

        samples_list = os.listdir(args.data_path)
        # samples_list = [os.path.join(args.data_path, each_sample, 'hcc_surg_artery_B.nii') for each_sample in samples_list if os.path.exists(os.path.join(args.data_path, each_sample, 'hcc_surg_artery_B.nii'))]
        samples_list = [os.path.join(args.data_path, each_sample, each_sample+'_tmv.npy') for each_sample in samples_list if os.path.exists(os.path.join(args.data_path, each_sample, each_sample+'_tmv.npy'))]

        print('start')
        list_of_args = []
        for th in range(len(samples_list)):
            each_artery_label = samples_list[th]
            each_artery_label_id = each_artery_label.split('/')[-2]
            # if each_artery_label_id in maden_sample_list:
            #     continue
            list_of_args.append((each_artery_label, args.save_path, args.f_sz, args.sigma))

        print(len(list_of_args))
        p = Pool(args.num_thread)
        p.starmap(get_density_and_save, list_of_args)
        p.close()
        p.join()
    else:
        maybe_mkdir_p(args.vis_save_path)
        samples_list = os.listdir(args.vis_data_path)
        samples_list = [os.path.join(args.vis_data_path, each_sample) for each_sample in samples_list]
        for each_sample in samples_list:
            each_sample_id = each_sample.split('/')[-1][:-4]
            density = np.load(each_sample, 'r')
            # print(density.shape) #[1,c,h,w]
            density_ = np.sum(density, axis=(0,2,3))
            slice_th = np.argmax(density_)

            density_slice = density.squeeze()[slice_th]
            density_slice = 255*density_slice/np.max(density_slice)
            density_slice = density_slice.astype(np.uint8)
            heat_img = cv2.applyColorMap(density_slice, cv2.COLORMAP_JET)
            fname = each_sample_id+'_density_map.png'
            cv2.imwrite(os.path.join(args.vis_save_path, fname), heat_img)