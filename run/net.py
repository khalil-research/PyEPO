#!/usr/bin/env python
# coding: utf-8
"""
PyTorch nn model
"""
from torch import nn

class fcNet(nn.Module):
    """
    multi-layer fully connected neural network regression
    """
    def __init__(self, arch, softplus=False):
        super().__init__()
        layers = []
        for i in range(len(arch)-1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            if i < len(arch) - 2:
                layers.append(nn.ReLU())
        if softplus:
            layers.append(nn.Softplus(threshold=5))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
