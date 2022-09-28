#!/usr/bin/env python
# coding: utf-8
"""
Train with Black-box optimization function
"""

import time
import os

from tqdm import tqdm
import torch

import pyepo
from .utils import getDevice

def trainDBB(reg, model, optimizer, trainloader, testloader=None, lossfunc="r",
            epoch=50, processes=1, bb_lambd=10, l1_lambd=0, l2_lambd=0):
    """
    A function to train PyTorch nn with Black-box optimization function

    Args:
        reg (nn): PyTorch neural network regressor
        model (optModel): optimization model
        optimizer (optim): PyTorch optimizer
        trainloader (DataLoader): PyTorch DataLoader for train set
        testloader (DataLoader): PyTorch DataLoader for test set
        lossfunc (str): loss function to use, "r" for regret, "h" for Hamming distance
        epoch (int): number of training epochs
        processes: processes (int): number of processors, 1 for single-core, 0 for all of cores
        bb_lambd (float): Black-Box parameter for function smoothing
        l1_lambd (float): regularization weight of l1 norm
        l2_lambd (float): regularization weight of l2 norm
    """
    # check loss type
    assert lossfunc in ["r", "h"], "argument 'lossfunc' must be 'r' or 'h'"
    # use training data for test if no test data
    if testloader is None:
        testloader = trainloader
    # get device
    device = getDevice()
    reg.to(device)
    # training mode
    reg.train()
    # set black-box optimizer
    dbb = pyepo.func.blackboxOpt(model, lambd=bb_lambd, processes=processes)
    # set loss
    criterion = torch.nn.L1Loss()
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
            # black-box optimizer
            wp = dbb(cp)
            # loss
            if lossfunc == "r":
                # objective value
                zp = (wp * c).sum(1).view(-1, 1)
                # regret
                loss = criterion(zp, z)
            if lossfunc == "h":
                # Hamming distance
                loss = criterion(wp, w)
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
