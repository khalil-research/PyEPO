#!/usr/bin/env python
# coding: utf-8
"""
PyTorch nn model
"""
import torch
from torch import nn
from torchvision.models import resnet18

class fcNet(nn.Module):
    """
    multi-layer fully connected neural network regression
    """
    def __init__(self, arch):
        super().__init__()
        layers = []
        for i in range(len(arch)-1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            if i < len(arch) - 2:
                layers.append(nn.ReLU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

# build new ResNet18 with Max Pooling
class partialResNet(nn.Module):

    def __init__(self, k):
        super(partialResNet, self).__init__()
        # init resnet 18
        resnet = resnet18(weights=None)
        # first five layers of ResNet18
        self.conv1 = resnet.conv1
        self.bn = resnet.bn1
        self.relu = resnet.relu
        self.maxpool1 = resnet.maxpool
        self.block = resnet.layer1
        # conv to 1 channel
        self.conv2  = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False)
        # max pooling
        self.maxpool2 = nn.AdaptiveMaxPool2d((k,k))

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn(h)
        h = self.relu(h)
        h = self.maxpool1(h)
        h = self.block(h)
        h = self.conv2(h)
        out = self.maxpool2(h)
        # reshape for optmodel
        out = torch.squeeze(out, 1)
        out = out.reshape(out.shape[0], -1)
        return out
