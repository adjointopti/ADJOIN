import glob, argparse
import os
import cv2
import sys
import skimage.io as sio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default='/home/eckert/Datasets/texture_data/pub/Bricks', type=str)
parser.add_argument("--save_path", default='/home/eckert/Datasets/texture_data/ply_G2L/pub/Bricks', type=str)
parser.add_argument("--model", default='ply_G2L', type=str)   # 'ply_G2L' 'ply_JTG' 'ply_intrinsic3d'
opt = parser.parse_args()

color_paths = sorted(glob.glob(os.path.join(opt.data_dir, "*_color.png")))
depth_paths = sorted(glob.glob(os.path.join(opt.data_dir, "*_depth.npz")))
cam_paths = sorted(glob.glob(os.path.join(opt.data_dir, "*_pose.txt")))

if opt.model == 'ply_JTG' or opt.model == 'ply_G2L':
    if not os.path.exists(os.path.join(opt.save_path, 'images')):
        os.makedirs(os.path.join(opt.save_path, 'images'))
if opt.model == 'ply_intrinsic3d':
    if not os.path.exists(os.path.join(opt.save_path, 'rgbd')):
        os.makedirs(os.path.join(opt.save_path, 'rgbd'))

num = 0
for img_dir in color_paths:
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if opt.model == 'ply_JTG':
        sio.imsave(os.path.join(opt.save_path, 'images/%05d.jpg' % num), img)
    if opt.model == 'ply_G2L':
        sio.imsave(os.path.join(opt.save_path, 'images/color_%02d.jpg' % num), img)
    if opt.model == 'ply_intrinsic3d':
        sio.imsave(os.path.join(opt.save_path, 'rgbd/frame-%06d.color.png' % num), img)
    num += 1


num = 0
for d in depth_paths:
    depth = np.load(d)['arr_0']
    depth0 = np.rint(depth*1000).astype('uint32')   # meter to millimeter
    depth0 = Image.fromarray(depth0)
    if opt.model == 'ply_JTG':
        depth0.save(os.path.join(opt.save_path, 'images/%05d.png' % num))
    if opt.model == 'ply_G2L':
        depth0.save(os.path.join(opt.save_path, 'images/depth_%02d.png' % num))
    if opt.model == 'ply_intrinsic3d':
        depth0.save(os.path.join(opt.save_path, 'rgbd/frame-%06d.depth.png' % num))
    num += 1


if opt.model == 'ply_JTG':
    fp = open(os.path.join(opt.save_path, 'pose6.log'), 'w')
    lines_new = ""
    num = 0
    for c in cam_paths:
        lines_new += '%d %d %d\n' % (num, num, num+1)
        pose = np.loadtxt(c)
        c2w = np.linalg.inv(pose)
        l_p = list(c2w)
        for p in l_p:
            value = ['%.6f' % p[j] for j in range(4)]
            lines_new += '%s\n' % ' '.join(value)
        num += 1
    fp.write(lines_new)
    fp.close()
if opt.model == 'ply_G2L':
    num = 0
    for c in cam_paths:
        pose = np.loadtxt(c)
        r = pose[:3,:3].ravel()
        t = pose[:3,3]
        cam = np.zeros([12,])
        cam[:3] = t
        cam[3:] = r

        fp = open(os.path.join(opt.save_path, 'images/color_%02d.cam' % num), 'w')
        value = ['%.6f' % cam[j] for j in range(12)]
        lines_new = '%s' % ' '.join(value)
        fp.write(lines_new)
        fp.close()
        num += 1
if opt.model == 'ply_intrinsic3d':
    num = 0
    for c in cam_paths:
        pose = np.loadtxt(c)
        c2w = np.linalg.inv(pose)
        np.savetxt(os.path.join(opt.save_path, 'rgbd/frame-%06d.pose.txt' % num), c2w)
        num += 1
    intrin = np.loadtxt(opt.data_dir+'/intrinsic.txt').astype('float32')
    intrinsics = np.identity(4).astype('float32')
    intrinsics[0, 0] = intrin[0, 0]
    intrinsics[1, 1] = intrin[1, 1]
    intrinsics[0, 2] = intrin[0, 2]
    intrinsics[1, 2] = intrin[1, 2]
    lines = ''
    for i in range(4):
        value = ['%.3f' % intrinsics[i, j] for j in range(4)]
        lines += '%s\n' % ' '.join(value)
    fp = open(os.path.join(opt.save_path, 'rgbd/colorIntrinsics.txt'), 'w')
    fp.write(lines)
    fp.close()
    if os.path.exists(opt.data_dir+'/intrinsic_depth.txt'):
        intrin = np.loadtxt(opt.data_dir + '/intrinsic.txt').astype('float32')
        intrinsics = np.identity(4).astype('float32')
        intrinsics[0, 0] = intrin[0, 0]
        intrinsics[1, 1] = intrin[1, 1]
        intrinsics[0, 2] = intrin[0, 2]
        intrinsics[1, 2] = intrin[1, 2]
        lines = ''
        for i in range(4):
            value = ['%.3f' % intrinsics[i, j] for j in range(4)]
            lines += '%s\n' % ' '.join(value)
    fp = open(os.path.join(opt.save_path, 'rgbd/depthIntrinsics.txt'), 'w')
    fp.write(lines)
    fp.close()

name = opt.data_dir.split('/')[-1]
print('Finish model ' + name)
