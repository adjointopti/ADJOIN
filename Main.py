"""
Author: Eckert ZHANG
Date: 2021-11-16 01:35:20
LastEditTime: 2021-11-18 01:38:42
LastEditors: Eckert ZHANG
FilePath: /Texture_code_v1.6/Main_simple_time.py
Description: 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys, glob, cv2, torch, lpips, yaml
import argparse, datetime, time, pdb
import numpy as np
import torchvision.utils as vutils
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.dataset import TextureDataset
from utils.mesh_assess import mesh_assess
from models.discriminator import PatchDiscriminator
from models.generator import Generator
from utils.loss import GANLoss, LaplacianLoss0, neg_iou_loss, averaged_hausdorff_dis
from utils.visualizer import Visualizer
from utils.gaussian_blur_cov import GaussianBlurConv
from utils.logger import Logger
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency, chamfer_distance, mesh_edge_loss
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim
from shutil import copyfile


def load_config():
    import configargparse
    r"""
    load configs
    """
    parser = configargparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default=
        '../../../Datasets/texture_data/gen_data/0noised/data_noised_rt6/yosee',
        help='Path of scanned RGB images')
    parser.add_argument(
        "--data_type",
        type=str,
        default='RGBD',
        help="'RGB' or 'RGBD'",
    )
    parser.add_argument(
        "--obj_dir",
        type=str,
        default='../../../Datasets/texture_data/gen_data/shape/gen_shape_eg6',
        help='Path of .obj model')
    parser.add_argument(
        "--depth_dir",
        type=str,
        default=None,
        # '../../../Datasets/texture_data/gen_data/1noised/data_noised_ed6/yosee',
    )
    parser.add_argument("--obj_gt_dir", type=str, default=None)
    parser.add_argument("--pose_gt_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default='./output_debug')
    parser.add_argument("--gan_weight", default=1.0, type=float)
    parser.add_argument("--l1_gan_initial", default=10.0, type=float)
    parser.add_argument("--l1_weight", default=0.1, type=float)
    parser.add_argument("--IOU_weight", default=1.0, type=float)
    parser.add_argument("--depth_weight", default=1.0, type=float)
    parser.add_argument("--laplacian_weight", default=1.0, type=float)
    parser.add_argument("--edge_weight", default=1.0, type=float)
    parser.add_argument("--lr_G_T", type=float, default=1e-2)
    parser.add_argument("--lr_G_V", type=float, default=1e-4)
    parser.add_argument("--lr_G_C", type=float, default=1e-3)
    parser.add_argument("--lr_D", type=float, default=1e-4)
    parser.add_argument("--batchSize", type=int, default=5)
    parser.add_argument(
        "--opti_mode",
        type=str,
        default='alter_adapt',
        help="",
    )
    parser.add_argument("--opti_order", type=str, default='C/V/T')
    parser.add_argument(
        "--nThreads",
        type=int,
        default=4,
        help='number works in dataloads',
    )
    parser.add_argument(
        "--unit_epoch",
        type=int,
        default=50,
        help='the unit epoch No. in every round',
    )
    parser.add_argument(
        "--num_T_unit",
        type=int,
        default=1,
        help='the Max No. of unit epoch per round for Texture optimization')
    parser.add_argument(
        "--num_V_unit",
        type=int,
        default=1,
        help='the Max No. of unit epoch per round for Vertice optimization')
    parser.add_argument(
        "--num_C_unit",
        type=int,
        default=1,
        help='the Max No. of unit epoch per round for Camera pose optimization'
    )
    parser.add_argument(
        "--total_round_joint",
        type=int,
        default=3,
        help='total rounds for joint optimization',
    )
    parser.add_argument(
        "--TransView",
        action='store_true',
        help='decide if there needs to perform the view transformation')
    parser.add_argument("--n_layers_D", type=int, default=4)
    parser.add_argument("--name",
                        type=str,
                        default='',
                        help='the name of this experiment')
    parser.add_argument("--initialized", type=int, default=1, help="")
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--gan_mode",
                        type=str,
                        default='ls',
                        help='the mode of GAN')
    parser.add_argument("--threshold_strategy", type=float, default=1e-3)
    parser.add_argument("--patience_strategy", type=int, default=50)
    opt = parser.parse_args()
    return opt


def get_texture(texture_path):
    texture_img = cv2.imread(texture_path, cv2.IMREAD_COLOR)
    texture_img = texture_img[..., ::-1].copy()
    texture = torch.tensor(texture_img) / 255.  # From [0,255] to [0,1]
    return texture


def construct_meshes(mesh, texture_map):
    verts = mesh.verts_list()[0]
    faces_ids = mesh.faces_list()[0]
    verts_uvs = mesh.textures.verts_uvs_list()[0]
    faces_uvs = mesh.textures.faces_uvs_list()[0]
    tex = TexturesUV(verts_uvs=[verts_uvs],
                     faces_uvs=[faces_uvs],
                     maps=texture_map)
    new_mesh = Meshes(verts=[verts], faces=[faces_ids], textures=tex)
    return new_mesh


def save_pose(save_path, pose, epoch, n_round):
    save_pose_path = os.path.join(
        save_path, 'training_pose/' + str(n_round) + '_' + str(epoch))
    if not os.path.exists(save_pose_path):
        os.makedirs(save_pose_path)
    tmp = pose.clone().cpu().detach()
    for i in range(tmp.shape[0]):
        np.savetxt(save_pose_path + '/%05d_pose.txt' % i, tmp[i].numpy())


def metric_render(opt, G, new_meshes, poses, state='initial'):
    dataset = TextureDataset(opt, opt.input_dir)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=opt.batchSize,
                                               shuffle=False,
                                               num_workers=opt.nThreads)
    # Measure Metrics
    psnr_list = []
    ssim_list = []
    perceptual_list = []
    # gradient_list = []
    loss_fn_alex = lpips.LPIPS(net='alex').to(opt.device)
    nn = 0
    for data_i in train_loader:
        nn += 1
        color_src_orig = data_i['color_src_orig'].to(opt.device)
        mask_src = data_i['mask_src'].to(opt.device)
        src_id = data_i['src_id']
        world2cam_src = poses[src_id].to(opt.device)
        mask3 = torch.cat((mask_src, mask_src, mask_src), dim=1)
        color_src_orig *= mask3

        render_results, render_mask, _, _, _ = G(world2cam_src, new_meshes)
        mask3_r = torch.cat((render_mask, render_mask, render_mask), dim=1)
        render_results *= mask3_r
        psnr_batch = []
        ssim_batch = []
        for ii in range(color_src_orig.shape[0]):
            img_gt = ((color_src_orig[ii, ...] + 1.0) * 0.5 *
                      255).cpu().detach().permute(1, 2,
                                                  0).numpy().astype('uint8')
            img_r = (np.clip(
                ((render_results[ii, ...] + 1.0) * 0.5).cpu().detach().permute(
                    1, 2, 0).numpy(), 0, 1) * 255).astype('uint8')
            val_psnr = ski_psnr(img_gt, img_r, data_range=255)
            val_ssim = ski_ssim(img_gt,
                                img_r,
                                data_range=255,
                                multichannel=True)
            psnr_batch.append(val_psnr)
            ssim_batch.append(val_ssim)

            if not os.path.exists(
                    os.path.join(opt.output_dir, 'Render_results')):
                os.makedirs(os.path.join(opt.output_dir, 'Render_results'))
            ## obtain white background
            mask = mask3[ii].cpu().permute(1, 2, 0).numpy().astype('uint8')
            mask_r = mask3_r[ii].cpu().detach().permute(
                1, 2, 0).numpy().astype('uint8')
            img_gt_w = img_gt * mask + 255 * (1 - mask)
            img_r_w = img_r * mask_r + 255 * (1 - mask_r)
            if state == 'initial':
                plt.imsave(
                    os.path.join(
                        opt.output_dir, 'Render_results/' + str(nn) + '_' +
                        str(ii + 1) + 'gt.png'), img_gt_w)
                plt.imsave(
                    os.path.join(
                        opt.output_dir, 'Render_results/' + str(nn) + '_' +
                        str(ii + 1) + 'render_input.png'), img_r_w)
            else:
                plt.imsave(
                    os.path.join(
                        opt.output_dir, 'Render_results/' + str(nn) + '_' +
                        str(ii + 1) + 'gt.png'), img_gt_w)
                plt.imsave(
                    os.path.join(
                        opt.output_dir, 'Render_results/' + str(nn) + '_' +
                        str(ii + 1) + 'render.png'), img_r_w)
        psnr_list.append(np.mean(psnr_batch))
        ssim_list.append(np.mean(ssim_batch))
        p_loss = loss_fn_alex(color_src_orig, render_results)
        perceptual_list.append(np.mean(p_loss.cpu().detach().numpy()))
    del loss_fn_alex

    return np.mean(psnr_list), np.mean(ssim_list), np.mean(perceptual_list)


def main(opt):
    torch.multiprocessing.set_start_method('spawn', force=True)
    tm0 = time.time()
    ### Create Dataset
    dataset = TextureDataset(opt, opt.input_dir)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=opt.batchSize,
                                               shuffle=True,
                                               num_workers=opt.nThreads)
    ### Define Model
    G = Generator(opt)
    D = PatchDiscriminator(opt)
    G = G.to(opt.device)
    D = D.to(opt.device)

    ### Load Mesh
    obj_name = opt.input_dir.split('/')[-1] + '.obj'
    meshes = load_objs_as_meshes([opt.obj_dir + '/' + obj_name],
                                 device=opt.device,
                                 load_textures=True)
    if opt.obj_gt_dir is not None and opt.initialized:
        mesh_gt = load_objs_as_meshes([opt.obj_gt_dir], load_textures=False)
        mesh_l1, mesh_l2 = mesh_assess(
            meshes.verts_normals_packed().cpu().detach(),
            mesh_gt.verts_normals_packed().cpu().detach())
        hd_aver = averaged_hausdorff_dis(
            meshes.verts_normals_packed().cpu().detach(),
            mesh_gt.verts_normals_packed().to(opt.device).cpu().detach())
        print(
            '\n\nOriginal Mesh:\nMesh_l1={:.5f}\nMesh_l2={:.5f}\nAverageHausdorff={:.5f}\n\n'
            .format(mesh_l1, mesh_l2, hd_aver))
    if opt.initialized:
        psnr_in, ssim_in, perceptual_in = metric_render(
            opt, G, meshes, dataset.world2cams.detach(), state='initial')
        print('For the initial rendered images:')
        print('PSNR_mean_initial = ', psnr_in)
        print('SSIM_mean_initial = ', ssim_in)
        print('perceptual_mean_initial = ', perceptual_in)
    deform_verts = torch.full(meshes.verts_packed().shape,
                              0.0,
                              device=opt.device,
                              requires_grad=True)
    deform_verts = torch.nn.Parameter(deform_verts)
    textures_map = torch.nn.Parameter(meshes.textures.maps_padded()[0])

    ### Load GT camera poses for comparison
    if opt.pose_gt_dir is not None and opt.initialized:
        cam_pose_gt = []
        cam_pose_gt_path = sorted(
            glob.glob(os.path.join(opt.pose_gt_dir, "*_pose.txt")))
        for pose_path in cam_pose_gt_path:
            w2c = np.loadtxt(pose_path)
            cam_pose_gt.append(w2c.astype('float32'))
        w2c_gt = torch.tensor(cam_pose_gt)
        pose_l1 = F.l1_loss(dataset.world2cams.clone(),
                            w2c_gt,
                            reduction='mean')
        print('\nOriginal pose_l1 = {:.5f}\n'.format(pose_l1))
        del cam_pose_gt, pose_path, cam_pose_gt_path

    ### Detect the order of optimization
    opti_order = opt.opti_order.split('/')
    node0 = eval('opt.num_' + opti_order[0] + '_unit')
    node1 = eval('opt.num_' + opti_order[1] + '_unit')
    node2 = eval('opt.num_' + opti_order[2] + '_unit')
    candidate_object = ['Texture', 'Vertice', 'CamPose']
    object_order = [0] * 3
    for o in candidate_object:
        index = opti_order.index(o[0])
        object_order[index] = o
    del candidate_object, index, opti_order, o

    ### Define optimizer and GAN Loss
    G_params = list(G.parameters())
    D_params = list(D.parameters())
    optimizer_G_V = torch.optim.Adam([deform_verts],
                                     lr=opt.lr_G_V,
                                     betas=(opt.beta1, opt.beta2))
    scheduler_G_V = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G_V,
        mode='min',
        factor=0.5,
        patience=20,
        verbose=True,
        threshold=5e-4,
        threshold_mode='rel',
        cooldown=20,
        min_lr=1e-6,
        eps=1e-7)
    optimizer_G_T = torch.optim.Adam([textures_map],
                                     lr=opt.lr_G_T,
                                     betas=(opt.beta1, opt.beta2))
    scheduler_G_T = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G_T,
        mode='min',
        factor=0.5,
        patience=20,
        verbose=True,
        threshold=1e-3,
        threshold_mode='rel',
        cooldown=20,
        min_lr=1e-5,
        eps=1e-6)
    optimizer_G_C = torch.optim.Adam([dataset.world2cams],
                                     lr=opt.lr_G_C,
                                     betas=(opt.beta1, opt.beta2))
    scheduler_G_C = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G_C,
        mode='min',
        factor=0.5,
        patience=40,
        verbose=True,
        threshold=1e-4,
        threshold_mode='rel',
        cooldown=40,
        min_lr=1e-5,
        eps=1e-6)
    optimizer_D = torch.optim.Adam(D_params,
                                   lr=opt.lr_D,
                                   betas=(opt.beta1, opt.beta2))
    criterionGAN = GANLoss(gan_mode=opt.gan_mode,
                           tensor=torch.cuda.FloatTensor)

    ### Define Visualizer
    visualize = Visualizer(opt)

    ### Mode of optimization and Preparing Calculation
    if (opt.num_T_unit + opt.num_V_unit + opt.num_C_unit) < 1 or (
            opt.num_T_unit * opt.num_V_unit * opt.num_C_unit) < 0:
        raise Exception("The num_unit of C/T/V should be Non-negative!")
    if opt.opti_mode == 'alter_adapt' or opt.opti_mode == 'alter_fix':
        round_epoch = (opt.num_T_unit + opt.num_V_unit +
                       opt.num_C_unit) * opt.unit_epoch
    elif opt.opti_mode == 'simultaneous' or opt.opti_mode == 'simul':
        round_epoch = max(opt.num_T_unit, opt.num_V_unit,
                          opt.num_C_unit) * opt.unit_epoch
    elif opt.opti_mode == 'alter_epoch':
        round_epoch = 3 * opt.unit_epoch
        object_order_repeat = object_order * round_epoch
    else:
        raise Exception("The opti_mode should be selected from expected mode!")

    total_epoch = opt.total_round_joint * round_epoch
    print('-' * 30 + "\n   Joint Optimization Begin!\n" + '-' * 30)
    print('The joint optimization mode is', opt.opti_mode)
    print('The total epochs of joint optimization is {}\n'.format(total_epoch))
    current_steps = 0
    G.train()
    D.train()
    for current_round in range(opt.total_round_joint):
        print('Current round of joint optimization = {}/{}'.format(
            current_round + 1, opt.total_round_joint))
        steps_per_round = 0

        if opt.opti_mode == 'alter_adapt':
            num_bad_epoch = 0
            los_best = 1
            los_current = 0
            thres_strategy = opt.threshold_strategy
            patience_strategy = opt.patience_strategy
            strategy = 'keep'
        for epoch in range(round_epoch):
            current_epoch = epoch + 1
            ### Strategy for Assignment of Optimization Object (part 1/2)
            if opt.opti_mode == 'alter_adapt':
                if current_epoch == node0 * opt.unit_epoch + 1 or current_epoch == (
                        node0 + node1) * opt.unit_epoch + 1:
                    num_bad_epoch = 0
                if num_bad_epoch >= patience_strategy and current_epoch > 5:
                    strategy = 'next'
                else:
                    strategy = 'keep'
                if current_epoch <= node0 * opt.unit_epoch:
                    if strategy == 'keep':
                        Opti_object = object_order[0]
                    else:
                        continue
                elif node0 * opt.unit_epoch < current_epoch <= (
                        node0 + node1) * opt.unit_epoch:
                    if strategy == 'keep':
                        Opti_object = object_order[1]
                    else:
                        continue
                else:
                    if strategy == 'keep':
                        Opti_object = object_order[2]
                    else:
                        continue
            elif opt.opti_mode == 'simultaneous' or opt.opti_mode == 'simul':
                Opti_object = 'All'
            elif opt.opti_mode == 'alter_epoch':
                Opti_object = object_order_repeat[epoch]
            elif opt.opti_mode == 'alter_fix':
                if current_epoch <= node0 * opt.unit_epoch:
                    Opti_object = object_order[0]
                elif node0 * opt.unit_epoch < current_epoch <= (
                        node0 + node1) * opt.unit_epoch:
                    Opti_object = object_order[1]
                else:
                    Opti_object = object_order[2]

            ### Beginning of one epoch
            for data_i in train_loader:
                ### Prepare training
                current_steps += 1
                steps_per_round += 1
                psnr_batch = []
                ssim_batch = []
                color_src, color_tar, world2cam_src, mask_src, mask_tar = data_i['color_src'], \
                    data_i['color_tar'], data_i['world2cam_src'], data_i['mask_src'], data_i['mask_tar']
                color_src = color_src.to(opt.device)
                color_tar = color_tar.to(opt.device)
                world2cam_src = world2cam_src.to(opt.device)
                mask_src = mask_src.to(opt.device)
                mask_tar = mask_tar.to(opt.device)
                color_src_orig = data_i['color_src_orig'].to(opt.device)
                mask3 = torch.cat((mask_tar, mask_tar, mask_tar), dim=1)
                ### Update Generator  ~1.10s
                if Opti_object == 'CamPose':
                    src_id = data_i['src_id']
                    world2cam_src = dataset.world2cams[src_id].to(
                        opt.device
                    )  # with gradient, different from the previous one.
                    optimizer_G_C.zero_grad()
                elif Opti_object == 'Texture':
                    optimizer_G_T.zero_grad()
                elif Opti_object == 'Vertice':
                    optimizer_G_V.zero_grad()
                elif Opti_object == 'All':
                    src_id = data_i['src_id']
                    world2cam_src = dataset.world2cams[src_id].to(
                        opt.device
                    )  # with gradient, different from the previous one.
                    optimizer_G_T.zero_grad()
                    optimizer_G_V.zero_grad()
                    optimizer_G_C.zero_grad()
                new_meshes = meshes.offset_verts(deform_verts)  # update vertex
                new_meshes = construct_meshes(
                    new_meshes, textures_map[None])  # update texture

                render_results, render_mask, render_silhouette, render_depth, depth_mask = G(
                    world2cam_src, new_meshes)

                ## Optimization
                render_results *= mask3
                if Opti_object == 'Texture':  # GAN loss is only calculated when Texture is optimized SEPARATELY
                    L1_weight = float(opt.l1_gan_initial) * (float(0.8)**float(
                        steps_per_round // 960))
                    G_L1 = F.l1_loss(
                        render_results, color_tar, reduction='none') * mask_tar
                    G_L1 = torch.sum(G_L1) / (torch.sum(mask_tar) * 3)

                    G_discriminate_input = torch.cat(
                        [color_src, render_results - color_src], dim=1)
                    D_output, D_mask = D(G_discriminate_input, mask_tar)
                    D_hard_mask = D_mask > 0.1
                    G_D_original = criterionGAN(
                        D_output, True, for_discriminator=False) * D_hard_mask
                    G_D = torch.sum(G_D_original) / (torch.sum(D_hard_mask) *
                                                     3)

                    IOU_loss = neg_iou_loss(render_silhouette, mask_src)

                    Generator_Loss = G_L1 * L1_weight + G_D
                    if opt.data_type == 'RGBD':
                        Depth_loss = F.l1_loss(
                            render_depth,
                            data_i['depth_src'].to(opt.device),
                            reduction='none') * depth_mask[:, 0, ...]
                        Depth_loss = torch.sum(Depth_loss) / torch.sum(
                            depth_mask[:, 0, ...])
                        common_loss = opt.l1_weight * G_L1 + opt.depth_weight * Depth_loss + opt.IOU_weight * IOU_loss
                    else:
                        common_loss = opt.l1_weight * G_L1 + opt.IOU_weight * IOU_loss
                    metric_reduce_lr_T_L1 = G_L1.clone()
                    los_current = common_loss.clone().cpu().detach()
                elif Opti_object == 'Vertice':
                    if current_round == 0:
                        gaussian_conv = GaussianBlurConv(
                            device=opt.device)  ## Gaussian Blur
                        blur_render = gaussian_conv(render_results) * mask3
                        blur_tar = gaussian_conv(color_tar) * mask3
                    else:
                        blur_render = render_results
                        blur_tar = color_tar

                    G_L1 = F.l1_loss(blur_render, blur_tar,
                                     reduction='none') * mask_tar
                    G_L1 = torch.sum(G_L1) / (torch.sum(mask_tar) * 3)

                    IOU_loss = neg_iou_loss(render_silhouette, mask_src)
                    lapla_L = mesh_laplacian_smoothing(
                        new_meshes, method='cot')  #'uniform' "cot"
                    if np.isnan(lapla_L.cpu().detach().numpy()):
                        print('NAN lap in step', current_steps)
                        break
                    edge_L = mesh_edge_loss(new_meshes)
                    if opt.data_type == 'RGBD':
                        Depth_loss = F.l1_loss(
                            render_depth,
                            data_i['depth_src'].to(opt.device),
                            reduction='none') * depth_mask[:, 0, ...]
                        Depth_loss = torch.sum(Depth_loss) / torch.sum(
                            depth_mask[:, 0, ...])
                        Generator_Loss = opt.l1_weight * G_L1 + opt.depth_weight * Depth_loss + opt.IOU_weight * IOU_loss \
                            + opt.laplacian_weight * lapla_L+ opt.edge_weight * edge_L
                        common_loss = opt.l1_weight * G_L1 + opt.depth_weight * Depth_loss + opt.IOU_weight * IOU_loss
                    else:
                        Generator_Loss = opt.laplacian_weight * lapla_L + opt.l1_weight * G_L1 + opt.IOU_weight * IOU_loss + opt.edge_weight * edge_L
                        common_loss = opt.l1_weight * G_L1 + opt.IOU_weight * IOU_loss
                    if steps_per_round % 10 == 0 and opt.data_type == 'RGBD':
                        print(
                            'color_loss = {:.5f}, depth_loss = {:.5f}, laplacian_loss = {:.5f}, IOU_loss = {:.5f}, edge_loss = {:.5f}'
                            .format(G_L1, Depth_loss, lapla_L, IOU_loss,
                                    edge_L))
                    metric_reduce_lr_V = Generator_Loss.clone()
                    los_current = common_loss.clone().cpu().detach()
                elif Opti_object == 'CamPose':
                    if current_round == 0:
                        ## Gaussian Blur
                        gaussian_conv = GaussianBlurConv(device=opt.device)
                        blur_render = gaussian_conv(render_results) * mask3
                        blur_src_orig = gaussian_conv(color_tar) * mask3
                    else:
                        blur_render = render_results
                        blur_src_orig = color_tar

                    G_L1 = F.l1_loss(blur_render,
                                     blur_src_orig,
                                     reduction='none') * mask_src
                    G_L1 = torch.sum(G_L1) / (torch.sum(mask_src) * 3)

                    IOU_loss = neg_iou_loss(render_silhouette, mask_src)
                    if opt.data_type == 'RGBD':
                        Depth_loss = F.l1_loss(
                            render_depth,
                            data_i['depth_src'].to(opt.device),
                            reduction='none') * depth_mask[:, 0, ...]
                        Depth_loss = torch.sum(Depth_loss) / torch.sum(
                            depth_mask[:, 0, ...])
                        Generator_Loss = opt.l1_weight * G_L1 + opt.depth_weight * Depth_loss + opt.IOU_weight * IOU_loss
                        common_loss = opt.l1_weight * G_L1 + opt.depth_weight * Depth_loss + opt.IOU_weight * IOU_loss
                    else:
                        Generator_Loss = opt.l1_weight * G_L1 + opt.IOU_weight * IOU_loss
                        common_loss = opt.l1_weight * G_L1 + opt.IOU_weight * IOU_loss
                    metric_reduce_lr_C = Generator_Loss.clone()
                    los_current = common_loss.clone().cpu().detach()
                elif Opti_object == 'All':
                    G_L1 = F.l1_loss(
                        render_results, color_tar, reduction='none') * mask_tar
                    G_L1 = torch.sum(G_L1) / (torch.sum(mask_tar) * 3)
                    IOU_loss = neg_iou_loss(render_silhouette, mask_src)
                    lapla_L = mesh_laplacian_smoothing(new_meshes,
                                                       method='uniform')
                    edge_L = mesh_edge_loss(new_meshes)
                    if opt.data_type == 'RGBD':
                        Depth_loss = F.l1_loss(
                            render_depth,
                            data_i['depth_src'].to(opt.device),
                            reduction='none') * depth_mask[:, 0, ...]
                        Depth_loss = torch.sum(Depth_loss) / torch.sum(
                            depth_mask[:, 0, ...])
                        Generator_Loss = opt.l1_weight * G_L1 + opt.IOU_weight * IOU_loss + opt.laplacian_weight * lapla_L \
                            + opt.depth_weight * Depth_loss + opt.edge_weight * edge_L
                        common_loss = opt.l1_weight * G_L1 + opt.depth_weight * Depth_loss + opt.IOU_weight * IOU_loss
                    else:
                        Generator_Loss = opt.l1_weight * G_L1 + opt.IOU_weight * IOU_loss + opt.laplacian_weight * lapla_L + opt.edge_weight * edge_L
                        common_loss = opt.l1_weight * G_L1 + opt.IOU_weight * IOU_loss
                    metric_reduce_lr_C = Generator_Loss.clone()
                    metric_reduce_lr_T_L1 = G_L1.clone()
                    metric_reduce_lr_V = Generator_Loss.clone()

                Generator_Loss.backward()
                if Opti_object == 'CamPose':
                    optimizer_G_C.step()
                    scheduler_G_C.step(metric_reduce_lr_C)
                if Opti_object == 'Vertice':
                    optimizer_G_V.step()
                    scheduler_G_V.step(metric_reduce_lr_V)
                if Opti_object == 'Texture':
                    optimizer_G_T.step()
                    scheduler_G_T.step(metric_reduce_lr_T_L1)
                if Opti_object == 'All':
                    optimizer_G_C.step()
                    optimizer_G_V.step()
                    optimizer_G_T.step()
                    scheduler_G_C.step(metric_reduce_lr_C)
                    scheduler_G_V.step(metric_reduce_lr_V)
                    scheduler_G_T.step(metric_reduce_lr_T_L1)

                ### Strategy for Assignment of Optimization Object (part 2/2)
                if opt.opti_mode == 'alter_adapt':
                    if current_epoch == 1 or current_epoch == node0 * opt.unit_epoch + 1 or current_epoch == (
                            node0 + node1) * opt.unit_epoch + 1:
                        los_best = los_current
                    if los_current < los_best * (1 - thres_strategy):
                        num_bad_epoch = 0
                        los_best = los_current
                    else:
                        num_bad_epoch += 1

                ### Update Discriminator   ~0.54s
                if Opti_object == 'Texture':
                    optimizer_D.zero_grad()
                    render_results, _, _, _, _ = G(world2cam_src, new_meshes)
                    render_results *= mask3
                    render_results = render_results.detach()
                    D_discriminate_input_real = torch.cat(
                        [color_src, color_tar - color_src], dim=1)
                    D_discriminate_input_fake = torch.cat(
                        [color_src, render_results - color_src], dim=1)
                    D_output_real, _ = D(D_discriminate_input_real, mask_tar)
                    D_output_fake, _ = D(D_discriminate_input_fake, mask_tar)
                    D1 = criterionGAN(D_output_real,
                                      True,
                                      for_discriminator=True) * D_hard_mask
                    D2 = criterionGAN(D_output_fake,
                                      False,
                                      for_discriminator=True) * D_hard_mask
                    D1_loss = torch.sum(D1) / (torch.sum(D_hard_mask) * 3)
                    D2_loss = torch.sum(D2) / (torch.sum(D_hard_mask) * 3)
                    Discriminator_Loss = (D1_loss + D2_loss) / 2
                    Discriminator_Loss.backward()
                    optimizer_D.step()
    tm2 = time.time()
    print('Total Time of Joint Optimization is',
          datetime.timedelta(seconds=round(tm2 - tm0)))
    del color_src, color_tar, world2cam_src, mask_src, mask_tar, D, render_results, render_mask, render_silhouette, render_depth, depth_mask, meshes

    ### Save & Measure Metrics
    G.save_texture(opt.output_dir, current_epoch, textures_map)
    G.save_mesh(opt.output_dir, current_epoch, new_meshes)
    save_pose(opt.output_dir, dataset.world2cams, current_epoch,
              current_round + 1)
    psnr_m, ssim_m, perceptual_m = metric_render(opt,
                                                 G,
                                                 new_meshes,
                                                 dataset.world2cams.detach(),
                                                 state='final')
    print('Finally, for the rendered images:')
    print('PSNR_mean = ', psnr_m)
    print('SSIM_mean = ', ssim_m)
    print('perceptual_mean = ', perceptual_m)

    ### Copy .mtl and new texture map
    mtl_name = opt.input_dir.split('/')[-1] + '.obj.mtl'
    mtl_path = os.path.join(opt.obj_dir, mtl_name)
    if os.path.exists(mtl_path):
        copyfile(mtl_path, opt.output_dir + '/training_mesh/' + mtl_name)
    else:
        mtl_name = opt.input_dir.split('/')[-1] + '.mtl'
        copyfile(os.path.join(opt.obj_dir, mtl_name),
                 opt.output_dir + '/training_mesh/' + mtl_name)
    copyfile(
        os.path.join(opt.output_dir,
                     "training_textures/%s.png" % str(current_epoch)),
        opt.output_dir +
        '/training_mesh/%s.png' % opt.input_dir.split('/')[-1])


if __name__ == '__main__':
    opt = load_config()
    opt.rotation_z = True
    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    color_image_path = sorted(
        glob.glob(os.path.join(opt.input_dir, "*_color.png")))
    if os.path.exists(color_image_path[0]):
        height, width, _ = cv2.imread(color_image_path[0]).shape
    else:
        raise Exception("Can not obtain the size of input image!")
    opt.hw = {'height': height, 'width': width}
    del color_image_path, height, width

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    sys.stdout = Logger(os.path.join(opt.output_dir, 'print_log.txt'))
    print('=============== config information ===============')
    for keys, value in vars(opt).items():
        print('{key}: {value}'.format(key=keys, value=value))
    print('=' * 50 + '\n')

    main(opt)
    print('Finish!')
