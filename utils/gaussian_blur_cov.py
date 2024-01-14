import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import random


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, device='cpu'):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.device = device
        kernel_size_candidate = [7, 9, 11]
        kernel_size = random.sample(kernel_size_candidate, 1)[0]
        std = random.uniform(1., 10.)
        kernel1 = cv2.getGaussianKernel(kernel_size, std)
        kernel2 = cv2.getGaussianKernel(kernel_size, std)
        kernel = np.dot(kernel1, kernel2.T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = np.repeat(kernel, self.channels, axis=0).to(self.device)
        self.padding = int((kernel_size-1)/2)

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=self.padding, groups=self.channels)
        return x
