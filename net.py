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
    def __init__(self, arch):
        super().__init__()
        layers = []
        for i in range(len(arch)-1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
