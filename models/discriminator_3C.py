import cv2
import torch
from PIL import Image
import os, sys
import numpy as np 
import torch.nn.functional as F





# Defines the PatchGAN discriminator with the specified arguments.

class PatchDiscriminator(torch.nn.Module):


    def __init__(self, opt):
        super(PatchDiscriminator,self).__init__()
        self.opt = opt

        input_nc=3 #6
        nf=64
        sequence = [torch.nn.Conv2d(input_nc, nf, kernel_size=4, stride=2, padding=1),
                     torch.nn.LeakyReLU(0.2, False)]
        mask_sequence=[torch.nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1)]

        for n in range(1, opt.n_layers_D): ## 4
            nf_prev = nf
            nf = min(nf * 2, 128)  # 512
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [torch.nn.Conv2d(nf_prev, nf, kernel_size=4, stride=stride, padding=1),torch.nn.LeakyReLU(0.2, False)]
            mask_sequence+=[torch.nn.Conv2d(1, 1, kernel_size=4, stride=stride, padding=1)]

        sequence += [torch.nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1)]
        mask_sequence+=[torch.nn.Conv2d(1, 1, kernel_size=4, stride=1, padding=1)]
        self.D=torch.nn.Sequential(*sequence)
        self.D_mask=torch.nn.Sequential(*mask_sequence)

        for layer in self.D:
            if type(layer)==torch.nn.Conv2d:
                torch.nn.init.normal_(layer.weight,0,0.02)
        
        for layer in self.D_mask:
            torch.nn.init.constant_(layer.weight,1/16.0)


    def forward(self, x, mask):

        output=self.D(x)
        mask_output=self.D_mask(mask)

        return output,mask_output

    