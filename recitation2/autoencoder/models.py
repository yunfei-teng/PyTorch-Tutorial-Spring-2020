# PyTorch tutorial codes for course Advanced Machine Learning
# models.py: define model structures
# read: https://pytorch.org/docs/stable/nn.html

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, use_unet):
        super().__init__()
        nf = 16
        self.use_unet = use_unet
        # convolution
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, nf, 4, 2, 1),
                        nn.BatchNorm2d(nf),
                        nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(nf, nf*2, 4, 2, 1),
                        nn.BatchNorm2d(nf*2),
                        nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
                        nn.Conv2d(nf*2, nf*4, 4, 2, 1),
                        nn.BatchNorm2d(nf*4),
                        nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
                        nn.Conv2d(nf*4, nf*8, 4, 2, 1),
                        nn.BatchNorm2d(nf*8),
                        nn.ReLU(True)
        )

        # transposed convolution
        self.trans_conv1 = nn.Sequential(
                        nn.ConvTranspose2d(nf, 3, 4, 2, 1),
        )
        self.trans_conv2 = nn.Sequential(
                        nn.ConvTranspose2d(nf*2, nf, 4, 2, 1),
                        nn.BatchNorm2d(nf),
                        nn.ReLU(True)
        )
        self.trans_conv3 = nn.Sequential(
                        nn.ConvTranspose2d(nf*4, nf*2, 4, 2, 1),
                        nn.BatchNorm2d(nf*2),
                        nn.ReLU(True)
        )
        self.trans_conv4 = nn.Sequential(
                        nn.ConvTranspose2d(nf*8, nf*4, 4, 2, 1),
                        nn.BatchNorm2d(nf*4),
                        nn.ReLU(True)
        )
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        if self.use_unet:
            y4 = self.trans_conv4(x4) + x3
            y3 = self.trans_conv3(y4) + x2
            y2 = self.trans_conv2(y3) + x1
            y1 = self.trans_conv1(y2)
        else:
            y4 = self.trans_conv4(x4)
            y3 = self.trans_conv3(y4)
            y2 = self.trans_conv2(y3)
            y1 = self.trans_conv1(y2)
        return torch.tanh(y1)
