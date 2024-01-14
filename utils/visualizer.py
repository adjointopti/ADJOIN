import os
import ntpath
import time
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch
import numpy as np


class Visualizer():
    def __init__(self, opt):
        self.opt = opt

        self.name = opt.name

        self.writer=SummaryWriter(log_dir=os.path.join(opt.output_dir,opt.name))        

        self.log_name = os.path.join(opt.output_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        self.writer.add_scalar('Loss/L1',errors['L1'].float(), step)
        if 'L2' in errors:
            self.writer.add_scalar('Loss/L2',errors['L2'].float(), step)
        if 'batch_psnr' in errors and 'batch_ssim' in errors:
            self.writer.add_scalar('Criterion/PSNR',errors['batch_psnr'], step)
            self.writer.add_scalar('Criterion/SSIM',errors['batch_ssim'], step)
        self.writer.add_scalar('LearningRate/lr_v',errors['lr_v'], step)
        self.writer.add_scalar('LearningRate/lr_t',errors['lr_t'], step)
        self.writer.add_scalar('LearningRate/lr_c',errors['lr_c'], step)
        if 'mesh_l1' in errors:
            self.writer.add_scalar('Mesh/mesh_l1', errors['mesh_l1'].float(), step)
            self.writer.add_scalar('Mesh/mesh_l2', errors['mesh_l2'].float(), step)
        if 'hausdorff' in errors:
            self.writer.add_scalar('Mesh/hausdorff', errors['hausdorff'], step)
        if 'Depth_loss' in errors:
            self.writer.add_scalar('Loss/Depth_loss', errors['Depth_loss'].float(), step)
        if 'IOU_loss' in errors:
            self.writer.add_scalar('Loss/IOU', errors['IOU_loss'].float(), step)
        if 'laplacian_loss' in errors:
            self.writer.add_scalar('Loss/laplacian', errors['laplacian_loss'].float(), step)
        if 'GAN' in errors:
            self.writer.add_scalars('Loss/GAN',{'G':errors['GAN'].float(),
            'D':(errors['D_fake'].float()+errors['D_real'].float())/2}, step)
        if 'pose_l1' in errors:
            self.writer.add_scalar('CamPose/pose_l1', errors['pose_l1'], step)
        with open(self.log_name, "a") as log_file:
            log_file.write('step:%s, %s\n' % (step,errors))
