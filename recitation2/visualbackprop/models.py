# PyTorch tutorial codes for course Advanced Machine Learning
# models.py: define model structures
# read: https://pytorch.org/docs/stable/nn.html
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out + x
        
class ConvNet(nn.Module):
    def __init__(self, use_batch_norm, use_resnet):
        super().__init__()
        nf = 16
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, etc.)
        # resized input size = 128
        m_layer = int(math.log2(128) - 2)
        self.net = nn.ModuleList()
        self.net.extend([nn.Conv2d(3, nf, 7, 1, 3, bias=False)])
        pre_mul, cur_mul = 1, 1
        
        for i in range(m_layer):
            cur_mul = min(2** i, 4)
            _net = []
            if use_resnet:
                _net += [ResidualBlock(nf* pre_mul,nf* pre_mul)]
            _net += [nn.Conv2d(nf* pre_mul, nf* cur_mul, 4, 2, 1, bias=False)]
            if use_batch_norm:
                _net += [nn.BatchNorm2d(nf* cur_mul)]
            _net += [nn.ReLU(True)]
            net = nn.Sequential(*_net)
            self.net.extend([net])
            pre_mul = cur_mul
        self.net.extend([nn.Conv2d(nf* cur_mul, 10, 4, 1, 0, bias=False)]) # 1     

    def forward(self, x):
        for idx, layer in enumerate(self.net):
            x = layer(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

