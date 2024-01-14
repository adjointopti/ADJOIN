import glob, argparse
import os, torch, pdb
import cv2
import skimage.io as sio
import numpy as np
import pickle
import sys
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader
)
import matplotlib.pyplot as plt
libpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(libpath + '/..')
sys.path.insert(1, '..')
from models.generator import OpenGLRealPerspectiveCameras, SimpleShader


class DRender2(torch.nn.Module):
    ## Render the image from a given mesh and world2cam matrix
    def __init__(self, opt):
        super(DRender2, self).__init__()
        self.opt = opt
        self.device = opt.device
        self.intrinsic = np.loadtxt(opt.camera_path + '/intrinsic.txt')
        self.intrinsic = np.reshape(self.intrinsic, [16])
        if os.path.exists(opt.camera_path + '/intrinsic_depth.txt'):
            self.render_d = True
            self.intrinsic_d = np.loadtxt(opt.camera_path + '/intrinsic_depth.txt')
            self.intrinsic_d = np.reshape(self.intrinsic_d, [16])
            if opt.hw_d:
                self.height_d = self.opt.hw_d['height']
                self.width_d = self.opt.hw_d['width']
            else:
                depth_src = np.load(opt.camera_path + '/00000_depth.npz')['arr_0']
                self.height_d, self.width_d = depth_src.shape
        else:
            self.render_d = False

    def forward(self, input_w2c, new_meshes):
        width = self.opt.hw['width']
        height = self.opt.hw['height']
        b = input_w2c.shape[0]
        Rz = torch.tensor([[-1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=torch.float32)
        Rz = Rz.view(1, 4, 4).repeat(b, 1, 1).to(self.device)
        input_w2c = torch.bmm(Rz, input_w2c)
        R = input_w2c[:, :3, :3].transpose(1, 2)
        T = input_w2c[:, :3, 3]

        # camera intrinsic
        edge_len = max(width, height)
        cameras = OpenGLRealPerspectiveCameras(
            device=self.device, focal_length=self.intrinsic[0],
            principal_point=((self.intrinsic[2], self.intrinsic[6]),), R=R,
            T=T, w=edge_len, h=edge_len, x0=0, y0=0)
        raster_settings = RasterizationSettings(
            image_size=edge_len,
            blur_radius=0.0,
            faces_per_pixel=1, )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SimpleShader(
                device=self.device
            )
        )
        images = renderer(new_meshes.extend(b))
        if width < height:
            images = images[:, :, height - width:, :]
        elif width > height:
            images = images[:, :height, :, :]
        color_render = images[:, ..., :3]
        mask_render = images[:, ..., 3:4].clone()
        mask_render[mask_render != 0] = 1
        del renderer, cameras, raster_settings

        if self.render_d:
            edge_len_d = max(self.height_d, self.width_d)
            cameras = OpenGLRealPerspectiveCameras(
                device=self.device, focal_length=self.intrinsic_d[0],
                principal_point=((self.intrinsic_d[2], self.intrinsic_d[6]),), R=R,
                T=T, w=edge_len_d, h=edge_len_d, x0=0, y0=0)
            raster_settings = RasterizationSettings(
                image_size=edge_len_d,
                blur_radius=0.0,
                faces_per_pixel=1, )
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=SimpleShader(
                    device=self.device,
                )
            )
            images_d = renderer(new_meshes.extend(b))
            if self.width_d < self.height_d:
                images_d = images_d[:, :, self.height_d - self.width_d:, :]
            elif self.width_d > self.height_d:
                images_d = images_d[:, :self.height_d, :, :]
            mask_d = images_d[:, ..., 3:4].clone()
            mask_d[mask_d != 0] = 1
            del renderer, cameras, raster_settings

        if not self.render_d:
            return color_render, mask_render   # BHWC
        else:
            return color_render, mask_render, mask_d   # BHWC


parser = argparse.ArgumentParser()
parser.add_argument("--color_path", default='/home/eckert/Datasets/data_original/intrinsic3d_original/hieroglyphics/rgbd', type=str)
parser.add_argument("--obj_path", default=None, type=str)
parser.add_argument("--depth_path", default=None, type=str)
parser.add_argument("--camera_path", default=None, type=str)
parser.add_argument("--save_path", default='/home/eckert/Datasets/texture_data/pub/hieroglyphics', type=str)
parser.add_argument("--model", default='Intrinsic3d', type=str)   # 'Fountain' 'Bolster' 'Intrinsic3d' 'Scene3D'
parser.add_argument("--keyframe_file", default=None, type=str)
opt = parser.parse_args()
opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if opt.depth_path is None:
    opt.depth_path = opt.color_path
if opt.camera_path is None:
    opt.camera_path = opt.color_path
if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

sparese_mask = True

# camera intrinsic
intrinsic = np.identity(4)
intrinsic_depth = np.identity(4)
if opt.model == 'Bolster':
    intrinsic[0,0] = 1081.37     # fx
    intrinsic[1,1] = 1081.37     # fy
    intrinsic[0,2] = 959.5     # cx
    intrinsic[1,2] = 539.5     # cy

    intrinsic_depth[0,0] = 540.69     # fx
    intrinsic_depth[1,1] = 540.69     # fy
    intrinsic_depth[0,2] = 479.75     # cx
    intrinsic_depth[1,2] = 269.75     # cy
elif opt.model == 'Fountain':
    intrinsic[0,0] = 1050.0     # fx
    intrinsic[1,1] = 1050.0     # fy
    intrinsic[0,2] = 639.5     # cx
    intrinsic[1,2] = 511.5     # cy

    intrinsic_depth[0,0] = 525.0     # fx
    intrinsic_depth[1,1] = 525.0     # fy
    intrinsic_depth[0,2] = 319.5     # cx
    intrinsic_depth[1,2] = 239.5     # cy
elif opt.model == 'Scene3D':
    intrinsic[0,0] = 525.0     # fx
    intrinsic[1,1] = 525.0     # fy
    intrinsic[0,2] = 319.5     # cx
    intrinsic[1,2] = 239.5     # cy

    intrinsic_depth[0,0] = 525.0     # fx
    intrinsic_depth[1,1] = 525.0     # fy
    intrinsic_depth[0,2] = 319.5     # cx
    intrinsic_depth[1,2] = 239.5     # cy
elif opt.model == 'Intrinsic3d':
    intrinsic[0,0] = 1170.19     # fx
    intrinsic[1,1] = 1170.19     # fy
    intrinsic[0,2] = 647.75     # cx
    intrinsic[1,2] = 483.75     # cy

    intrinsic_depth[0,0] = 577.871     # fx
    intrinsic_depth[1,1] = 580.258     # fy
    intrinsic_depth[0,2] = 319.623     # cx
    intrinsic_depth[1,2] = 239.624     # cy
else:
    raise Exception('Input model is not recorded!!!')
np.savetxt(opt.save_path+'/intrinsic.txt', intrinsic)
np.savetxt(opt.save_path+'/intrinsic_depth.txt', intrinsic_depth)


# Key frames
key_record = None
if opt.model == 'Intrinsic3d' and opt.keyframe_file is not None:
    lines = [l.strip() for l in open(opt.keyframe_file)]
    key_record = []
    for i in range(1,len(lines)):
        l = lines[i]
        words = [w for w in l.split(' ') if w != '']
        key_record.append(int(words[1]))
    key_record = np.array(key_record)


## Camera pose
if opt.model == 'Bolster':
    pose_paths = sorted(glob.glob(os.path.join(opt.camera_path, "*.cam")))
    num_views = len(pose_paths)
    num = 0
    poses = []
    for p in pose_paths:
        cam = np.loadtxt(p)
        t = cam[:3].reshape(3, 1)
        r = cam[3:].reshape(3, 3)
        w2c = np.identity(4)
        w2c[:3, :3] = r
        w2c[:3, 3:4] = t
        poses.append(w2c)
        np.savetxt(os.path.join(opt.save_path, '%05d_pose.txt' % num), w2c)
        num += 1
if opt.model == 'Fountain' or opt.model == 'Scene3D':
    lines = [l.strip() for l in open(opt.camera_path)]
    num_views = np.int((len(lines) / 5))
    poses = []
    for i in range(num_views):
        num_line = 5*i
        c2w = np.identity(4)
        for j in range(4):
            l = lines[num_line + j+1]
            words = [w for w in l.split(' ') if w != '']
            if len(words) == 1:
                words = [w for w in l.split('\t') if w != '']
            c2w[j,:] = np.array([float(words[0]), float(words[1]), float(words[2]), float(words[3])])
        w2c = np.linalg.inv(c2w)
        poses.append(w2c)
        np.savetxt(os.path.join(opt.save_path, '%05d_pose.txt' % i), w2c)
if opt.model == 'Intrinsic3d':
    pose_paths = sorted(glob.glob(os.path.join(opt.camera_path, "*.pose.txt")))
    if key_record is not None:
        poses_key = []
        for i in range(len(key_record)):
            if key_record[i] == 1:
                poses_key.append(pose_paths[i])
        pose_paths = poses_key
        del poses_key
    num_views = len(pose_paths)
    num = 0
    poses = []
    for p in pose_paths:
        cam = np.loadtxt(p)
        w2c = np.linalg.inv(cam)
        poses.append(w2c)
        np.savetxt(os.path.join(opt.save_path, '%05d_pose.txt' % num), w2c)
        num += 1


# Compute pose pair
del w2c
fp = open(os.path.join(opt.save_path, 'pose_pair.pkl'), 'wb')
t = []
min_len = 10000
for i in range(num_views):
    a = []
    for j in range(num_views):
        angle = np.dot(poses[i][:,2], poses[j][:,2])
        if angle > np.cos(15.0 / 180.0 * np.pi):
            a.append(j)
    if len(a) < min_len:
        min_len = len(a)
    t.append(a)
pickle.dump(t, fp)
fp.close()


# Color images & depth map & mask
if opt.model == 'Bolster' or opt.model == 'Fountain':
    color_paths = sorted(glob.glob(os.path.join(opt.color_path, "*.jpg")))
    depth_paths = sorted(glob.glob(os.path.join(opt.depth_path, "*.png")))
if opt.model == 'Scene3D':
    color_paths = sorted(glob.glob(os.path.join(opt.color_path, "*.png")))
    depth_paths = sorted(glob.glob(os.path.join(opt.depth_path, "*.png")))
if opt.model == 'Intrinsic3d':
    color_paths = sorted(glob.glob(os.path.join(opt.color_path, "*.color.png")))
    depth_paths = sorted(glob.glob(os.path.join(opt.depth_path, "*.depth.png")))
    if key_record is not None:
        paths_key = []
        paths_key_d = []
        for i in range(len(key_record)):
            if key_record[i] == 1:
                paths_key.append(color_paths[i])
                paths_key_d.append(depth_paths[i])
        color_paths = paths_key
        depth_paths = paths_key_d
        del paths_key, paths_key_d
# render mask or not
if opt.obj_path is not None:
    height, width, _ = cv2.imread(color_paths[0]).shape
    opt.hw = {'height': height, 'width': width}
    height_d, width_d = cv2.imread(depth_paths[0], cv2.IMREAD_UNCHANGED).shape
    opt.hw_d = {'height': height_d, 'width': width_d}
    opt.camera_path = opt.save_path
    mesh = load_objs_as_meshes([opt.obj_path], device=opt.device, load_textures=True)
    G = DRender2(opt)
    G = G.cuda()

if not len(depth_paths)==len(color_paths):
    raise Exception('The number of color image and the number of depth should be same!')
num = 0
for d in depth_paths:
    dep = cv2.imread(d, cv2.IMREAD_UNCHANGED)
    img = cv2.imread(color_paths[num], cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if opt.obj_path is not None:
        world2cam = torch.tensor(poses[num].astype('float32'))[None].to(opt.device)
        render_results, mask_c, mask_d = G(world2cam, mesh)
        mask = mask_c[0,:,:,0].detach().cpu().numpy().astype('uint8')
        mask_d = mask_d[0,:,:,0].detach().cpu().numpy().astype('uint8')
        for ii in range(3):
            img[...,ii] *= mask
        dep *= mask_d
        if sparese_mask:
            # pdb.set_trace()
            ratio = np.sum(mask) / mask.size
            if ratio <= 0.3:
                os.remove(os.path.join(opt.save_path, '%05d_pose.txt' % num))
                num += 1
                continue
    else:
        mask = img[:, :, 0].copy()
        mask[mask >= 0] = 1
    dep_meter = np.float32(dep)/1000.    # mm to m
    np.savez_compressed(os.path.join(opt.save_path, '%05d_depth.npz' % num), dep_meter)
    sio.imsave(os.path.join(opt.save_path, '%05d_mask.png' % num), (mask*255).astype('uint8'))
    sio.imsave(os.path.join(opt.save_path, '%05d_color.png' % num), img)
    num += 1

print('Final!')
