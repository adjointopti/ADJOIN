"""
Author: Eckert ZHANG
Date: 2021-11-16 01:34:43
LastEditTime: 2021-11-19 21:40:40
LastEditors: Eckert ZHANG
FilePath: /Texture_code_v1.6/utils/dataset.py
Description: 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import glob
import cv2
import numpy as np
import pickle
import random
import torch.utils.data as data
import torch
import matplotlib.pyplot as plt
import pylab


class TextureDataset(data.Dataset):
    def __init__(self, opt, parent_dir):
        super(TextureDataset, self).__init__()
        self.opt = opt
        if parent_dir is None or not os.path.exists(parent_dir):
            raise Exception("input_dir does not exist")

        ## Load training pairs and RGB images
        self.kernel = np.ones((11, 11), np.uint8)
        self.color_paths = sorted(
            glob.glob(os.path.join(parent_dir,
                                   "*_color.png")))  ## All the RGB images

        ## Load training paris
        self.view_pairs = pickle.load(open(parent_dir + '/pose_pair.pkl',
                                           'rb'))  ## The training use pairs
        for i in range(len(self.view_pairs)):
            if type(self.view_pairs[i]) == type([]):
                p = self.view_pairs[i].copy()
            else:
                p = self.view_pairs[i].tolist()
            p.append(i)
            self.view_pairs[i] = np.array(p, dtype='int32')

        ## Load camera intrinsic
        self.intrinsic = np.loadtxt(parent_dir + '/intrinsic.txt')
        self.intrinsic = np.reshape(self.intrinsic, [16])

        ### Load camera poses for optimization
        cam_pose = []
        cam_pose_path = sorted(
            glob.glob(os.path.join(parent_dir, "*_pose.txt")))
        for pose_path in cam_pose_path:
            w2c = np.loadtxt(pose_path)
            cam_pose.append(w2c.astype('float32'))
        self.world2cams = torch.nn.Parameter(torch.tensor(cam_pose))

    def LoadDataByID(self, index):
        # if not index in dictionary:
        color_src_img = self.opt.input_dir + '/%05d_color.png' % index
        mask_src_img = self.opt.input_dir + '/%05d_mask.png' % index
        color_src = cv2.imread(color_src_img) / 255.0
        mask_src = cv2.imread(mask_src_img, cv2.IMREAD_UNCHANGED) / 255.0
        if self.opt.depth_dir is None:
            depth_src_img = self.opt.input_dir + '/%05d_depth.npz' % index
        else:
            depth_src_img = self.opt.depth_dir + '/%05d_depth.npz' % index
        depth_src = np.load(depth_src_img)['arr_0']

        L = np.min(depth_src[depth_src > 0])
        U = np.max(depth_src[depth_src > 0])
        mask = depth_src != 0
        d_i = ((depth_src - L) / (U - L) * 255 * mask).astype('uint8')
        os.makedirs('./depth_file', exist_ok=True)
        plt.imsave(f'depth_file/depth_{index}.png', d_i)

        return color_src.astype('float32'), depth_src.astype('float32'), \
               mask_src.astype('float32')

    def __getitem__(self, index):
        fn = self.color_paths[index]
        _, fullname = os.path.split(fn)
        src_id = int(fullname[:-10])
        color_src, depth_src, mask_src = self.LoadDataByID(src_id)
        world2cam_src = self.world2cams[src_id].detach().numpy()
        color_src_orig = color_src.copy()

        IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL = color_src.shape

        if self.opt.TransView:
            random.seed(0)
            rindex = random.choice(self.view_pairs[src_id])
        else:
            rindex = src_id
        if rindex != src_id:
            color_tar, depth_tar, mask_tar = self.LoadDataByID(rindex)
            world2cam_tar = self.world2cams[rindex].detach().numpy()

            cam2world_src = np.linalg.inv(world2cam_src)
            src2tar = np.transpose(np.dot(world2cam_tar, cam2world_src))

            y = np.linspace(0, IMAGE_HEIGHT - 1, IMAGE_HEIGHT)
            x = np.linspace(0, IMAGE_WIDTH - 1, IMAGE_WIDTH)
            xx, yy = np.meshgrid(x, y)

            fx = self.intrinsic[0]
            cx = self.intrinsic[2]
            fy = self.intrinsic[5]
            cy = self.intrinsic[6]

            x = (xx - cx) / fx * depth_src
            y = (yy - cy) / fy * depth_src
            coords = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
            coords[:, :, 0] = x
            coords[:, :, 1] = y
            coords[:, :, 2] = depth_src
            coords[:, :, 3] = 1
            coords = np.dot(coords, src2tar)
            z_tar = coords[:, :, 2]
            x = coords[:, :, 0] / (1e-8 + z_tar) * fx + cx
            y = coords[:, :, 1] / (1e-8 + z_tar) * fy + cy

            mask0 = (depth_src == 0)
            mask1 = (x < 0) + (y < 0) + (x >= IMAGE_WIDTH - 1) + (
                y >= IMAGE_HEIGHT - 1)
            lx = np.floor(x).astype('float32')
            ly = np.floor(y).astype('float32')
            rx = (lx + 1).astype('float32')
            ry = (ly + 1).astype('float32')
            sample_z1 = np.abs(z_tar \
                               - cv2.remap(depth_tar, lx, ly, cv2.INTER_NEAREST))
            sample_z2 = np.abs(z_tar \
                               - cv2.remap(depth_tar, lx, ry, cv2.INTER_NEAREST))
            sample_z3 = np.abs(z_tar \
                               - cv2.remap(depth_tar, rx, ly, cv2.INTER_NEAREST))
            sample_z4 = np.abs(z_tar \
                               - cv2.remap(depth_tar, rx, ry, cv2.INTER_NEAREST))
            mask2 = np.minimum(np.minimum(sample_z1, sample_z2),
                               np.minimum(sample_z3, sample_z4)) > 0.1

            mask_remap = (1 - (mask0 + mask1 + mask2 > 0)).astype('float32')

            map_x = x.astype('float32')
            map_y = y.astype('float32')

            color_tar_to_src = cv2.remap(color_tar, map_x, map_y,
                                         cv2.INTER_LINEAR)
            ## for figure of paper (1/3)
            # color_tar_for_figure = color_tar_to_src.copy()
            # color_tar_for_figure[..., 2][color_tar_for_figure[..., 2] == 0] = 1
            # color_tar_for_figure[..., 0][color_tar_for_figure[..., 0] == 0] = 1

            mask = (cv2.remap(mask_tar, map_x, map_y, cv2.INTER_LINEAR) > 0.99) \
                   * mask_remap
            for j in range(3):
                color_tar_to_src[:, :, j] *= mask
        else:
            color_tar_to_src = color_src.copy()
            mask = mask_src.copy()

            ## for figure of paper (2/3)
            # color_tar_for_figure = color_tar_to_src.copy()

        ## for figure of paper (3/3)
        # for j in range(3):
        #     color_tar_for_figure[:, :, j] *= mask
        # plt.figure()
        # plt.imshow(cv2.cvtColor(color_tar_for_figure, cv2.COLOR_RGB2BGR))
        # plt.imsave('./reprojected_0.png', cv2.cvtColor(color_tar_for_figure, cv2.COLOR_RGB2BGR))
        # plt.show()

        color_src = (color_src * 2.0 - 1.0).astype('float32')
        color_tar_to_src = (color_tar_to_src * 2.0 - 1.0).astype('float32')
        color_src_orig = (color_src_orig * 2.0 - 1.0).astype('float32')

        for i in range(3):
            color_src[:, :, i] *= mask
            color_tar_to_src[:, :, i] *= mask
            color_src_orig[:, :, i] *= mask_src

        # if cached:
        #     dictionary[fn] = (color_src, color_tar_to_src, uv_src,\
        #         np.reshape(mask,(mask.shape[0],mask.shape[1],1)))
        #     return dictionary[fn]
        # plt.figure()
        # plt.imshow(mask_src)
        # plt.axis('off')
        # plt.show
        # plt.savefig('src.png')

        color_src_tensor = self.opencv2tensor(color_src)  # BGR2RGB && CHW
        color_tar_tensor = self.opencv2tensor(color_tar_to_src)  # CHW
        color_src_orig = self.opencv2tensor(color_src_orig)  # CHW
        mask_src_tensor = torch.tensor(mask_src).float()  # 1HW
        mask_tar_tensor = torch.tensor(mask).float()
        data_dict = {
            'color_src': color_src_tensor,
            'color_tar': color_tar_tensor,
            'mask_tar': mask_tar_tensor.unsqueeze(0),
            'mask_src': mask_src_tensor.unsqueeze(0),
            'world2cam_src': world2cam_src,
            'color_src_orig': color_src_orig,
            'src_id': src_id,
            'depth_src': torch.tensor(depth_src)
        }
        return data_dict

    def __len__(self):
        return len(self.color_paths)

    @staticmethod
    def opencv2tensor(cv2_img):
        img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        return img.float()

    @staticmethod
    def name():
        return 'Texture_Optimization_Dataset'
