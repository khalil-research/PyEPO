#!/usr/bin/env python
# coding: utf-8
"""
Train with SPO+ loss
"""

import time
import os

import pandas as pd
from tqdm import tqdm
import torch

import pyepo
from .utils import getDevice

def trainSPO(reg, model, optimizer, trainloader, testloader=None,
             epoch=50, processes=1, l1_lambd=0, l2_lambd=0):
    """
    A function to train PyTorch nn with SPO+ loss

    Args:
        reg (nn): PyTorch neural network regressor
        model (optModel): optimization model
        optimizer (optim): PyTorch optimizer
        trainloader (DataLoader): PyTorch DataLoader for train set
        testloader (DataLoader): PyTorch DataLoader for test set
        epoch (int): number of training epochs
        processes: processes (int): number of processors, 1 for single-core, 0 for all of cores
        l1_lambd (float): regularization weight of l1 norm
        l2_lambd (float): regularization weight of l2 norm
    """
    # use training data for test if no test data
    if testloader is None:
        testloader = trainloader
    # get device
    device = getDevice()
    reg.to(device)
    # training mode
    reg.train()
    # set SPO+ Loss as criterion
    spop = pyepo.func.SPOPlus(model, processes=processes)
    # train
    time.sleep(1)
    pbar = tqdm(range(epoch))
    cnt = 0
    trueloss = None
    unambloss = None
    for epoch in pbar:
        # load data
        for i, data in enumerate(trainloader):
            x, c, w, z = data
            x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
            # forward pass
            cp = reg(x)
            loss = spop(cp, c, w, z).mean()
            # l1 reg
            if l1_lambd:
                l1_reg = torch.abs(cp - c).mean(dim=1).mean()
                loss += l1_lambd * l1_reg
            # l2 reg
            if l2_lambd:
                l2_reg = ((cp - c) ** 2).mean(dim=1).mean()
                loss += l2_lambd * l2_reg
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            desc = "Epoch {}, Loss: {:.4f}".format(epoch, loss.item())
            pbar.set_description(desc)
            cnt += 1
