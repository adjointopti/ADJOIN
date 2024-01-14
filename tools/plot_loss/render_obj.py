import numpy as np
import torch
import sys
import os
import glob
import argparse
import cv2
import matplotlib.pyplot as plt
import lpips
import torch.nn as nn
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Textures, Meshes
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    PointLights,
    get_world_to_view_transform,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturedSoftPhongShader,
    SoftPhongShader
)
from typing import NamedTuple, Sequence
sys.path.insert(1, '..')
from models.generator import OpenGLRealPerspectiveCameras, SimpleShader
from utils.logger import Logger


class DRender_obj(torch.nn.Module):
    ## Render the image from a given mesh and world2cam matrix
    def __init__(self, opt):
        super(DRender_obj, self).__init__()
        self.opt = opt
        self.device = opt.device
        self.intrinsic = np.loadtxt(opt.cam_path + '/intrinsic.txt')
        self.intrinsic = np.reshape(self.intrinsic, [16])

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
        # lights = PointLights(location=[[0.0, 0.0, -3.0]],
        #                     ambient_color=((1, 1, 1),),
        #                     diffuse_color=((0, 0, 0),),
        #                     specular_color=((0, 0, 0),),
        #                     device=self.device,)
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
        # plt.figure()
        # plt.imshow(images[0, ..., :3].cpu().detach().numpy())
        # plt.axis('off')
        # plt.show()
        if width < height:
            images = images[:, :, height - width:, :]
        elif width > height:
            images = images[:, :height, :, :]
        color_render = images[:, ..., :3]
        mask_render = images[:, ..., 3:4].clone()
        mask_render[mask_render != 0] = 1

        return color_render, mask_render   # BCHW

class DRender_ply(torch.nn.Module):
    ## Render the image from a given mesh and world2cam matrix
    def __init__(self, opt):
        super(DRender_ply, self).__init__()
        self.opt = opt
        self.device = opt.device
        self.intrinsic = np.loadtxt(opt.cam_path + '/intrinsic.txt')
        self.intrinsic = np.reshape(self.intrinsic, [16])

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
        lights = PointLights(location=[[0.0, 0.0, -3.0]],
                            ambient_color=((1, 1, 1),),
                            diffuse_color=((0, 0, 0),),
                            specular_color=((0, 0, 0),),
                            device=self.device,)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights
            )
        )
        images = renderer(new_meshes.extend(b))
        # plt.figure()
        # plt.imshow(images[0, ..., :3].cpu().detach().numpy())
        # plt.axis('off')
        # plt.show()
        if width < height:
            images = images[:, :, height - width:, :]
        elif width > height:
            images = images[:, :height, :, :]
        color_render = images[:, ..., :3]
        mask_render = images[:, ..., 3:4].clone()
        mask_render[mask_render != 0] = 1

        return color_render, mask_render# BCHW


parser = argparse.ArgumentParser()
parser.add_argument("--obj_path", default='../../texture_data/ply_G2L/M20/194725/results/result_label.obj', type=str)  #G2LTex
parser.add_argument("--cam_path", default='../../texture_data/HW/new_M20/194725', type=str)
parser.add_argument("--imgGT_path", default='../../texture_data/ply_G2L/M20/194725/images', type=str)
parser.add_argument("--output_dir", default='../../texture_data/ply_G2L/M20/194725/results', type=str)
parser.add_argument("--M20", default=None, type=str)
opt = parser.parse_args()
opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if 'M20' in opt.obj_path:
    opt.M20 = 'M20'

cam_pose_path = sorted(glob.glob(os.path.join(opt.cam_path, "*_pose.txt")))
colors_path = sorted(glob.glob(os.path.join(opt.imgGT_path, "*.jpg")))
if len(colors_path) == 0:
    colors_path = sorted(glob.glob(os.path.join(opt.imgGT_path, "*_color.png")))
if os.path.exists(colors_path[0]):
    height, width, _ = cv2.imread(colors_path[0]).shape
else:
    raise Exception("Can not obtain the size of input image!")
opt.hw = {'height': height, 'width': width}
if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir)
sys.stdout = Logger(os.path.join(opt.output_dir, 'print_log.txt'))

mesh = load_objs_as_meshes([opt.obj_path], device=opt.device, load_textures=True)

G = DRender_obj(opt)
G = G.cuda()
psnr_list = []
ssim_list = []
perceptual_list = []
gradient_list = []
loss_fn_alex = lpips.LPIPS(net='alex').to(opt.device)
num = 0
for c in colors_path:
    img_gt = cv2.imread(c, cv2.IMREAD_UNCHANGED)
    img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
    img_gt0 = img_gt.copy()
    w2c = torch.tensor(np.loadtxt(cam_pose_path[num]).astype('float32'))[None].to(opt.device)

    render_results, render_mask = G(w2c, mesh)

    mask3_r = torch.cat((render_mask, render_mask, render_mask), dim=3).detach()
    if opt.M20 is not None:
        mask_img = img_gt0[:,:,0] + img_gt0[:,:,1] + img_gt0[:,:,2]
        mask_img[mask_img>0]=1
        for ii in range(3):
            mask3_r[0,:,:,ii] *= torch.tensor(mask_img).cuda()
    render_results *= mask3_r
    mask3_r = mask3_r[0].cpu().numpy()
    img_r = (render_results[0]*255).cpu().detach().numpy().astype('uint8')
    img_gt *= mask3_r.astype('uint8')

    # plt.figure()
    # plt.imshow(img_r)
    # plt.axis('off')
    # plt.show()
    val_psnr = ski_psnr(img_gt, img_r, data_range=255)
    val_ssim = ski_ssim(img_gt, img_r, data_range=255, multichannel=True)
    psnr_list.append(val_psnr)
    ssim_list.append(val_ssim)
    if not os.path.exists(os.path.join(opt.output_dir, 'Render_results')):
        os.makedirs(os.path.join(opt.output_dir, 'Render_results'))
    plt.imsave(os.path.join(opt.output_dir, 'Render_results/' + str(num) + 'gt.png'), img_gt0)
    plt.imsave(os.path.join(opt.output_dir, 'Render_results/' + str(num) + 'render.png'), img_r)
    # gradients of img
    gt_sobelx = cv2.Sobel(img_gt, cv2.CV_64F, 1, 0, 3)
    gt_sobely = cv2.Sobel(img_gt, cv2.CV_64F, 0, 1, 3)
    gt_sobel = cv2.addWeighted(cv2.convertScaleAbs(gt_sobelx), 0.5, cv2.convertScaleAbs(gt_sobely), 0.5, 0)
    r_sobelx = cv2.Sobel(img_r, cv2.CV_64F, 1, 0, 3)
    r_sobely = cv2.Sobel(img_r, cv2.CV_64F, 0, 1, 3)
    r_sobel = cv2.addWeighted(cv2.convertScaleAbs(r_sobelx), 0.5, cv2.convertScaleAbs(r_sobely), 0.5, 0)
    plt.imsave(os.path.join(opt.output_dir, 'Render_results/' + str(num) + 'gradient_gt.png'), gt_sobel)
    plt.imsave(os.path.join(opt.output_dir, 'Render_results/' + str(num) + 'gradient_r.png'), r_sobel)
    gradient_diff = np.mean(np.abs(gt_sobel - r_sobel)) / 255
    gradient_list.append(gradient_diff)
    # perceptual of img
    p_loss = loss_fn_alex(torch.tensor(img_gt.astype('float32')/255.*2.-1.)[None].permute(0,3,1,2).cuda(), (render_results*2-1).permute(0,3,1,2))
    perceptual_list.append(np.mean(p_loss.cpu().detach().numpy()))
    num += 1

print('Finally, for the rendered images:')
print('PSNR_mean = ', np.mean(psnr_list))
print('SSIM_mean = ', np.mean(ssim_list))
print('perceptual_mean = ', np.mean(perceptual_list))
print('gradient_mean = ', np.mean(gradient_list))

