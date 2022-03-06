#!/usr/bin/env python
# coding: utf-8
"""
Train with SPO+ loss
"""

import os
import time

import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

import pyepo
from pyepo.train.util import getDevice

def trainSPO(reg, model, optimizer, trainloader, testloader=None, logdir="./logs",
             epoch=50, processes=1, l1_lambd=0, l2_lambd=0, log=0):
    """
    A function to train PyTorch nn with SPO+ loss

    Args:
        reg (nn): PyTorch neural network regressor
        model (optModel): optimization model
        optimizer (optim): PyTorch optimizer
        trainloader (DataLoader): PyTorch DataLoader for train set
        testloader (DataLoader): PyTorch DataLoader for test set
        logdir (str): folder path to save tensorboard log
        epoch (int): number of training epochs
        processes: processes (int): number of processors, 1 for single-core, 0 for all of cores
        l1_lambd (float): regularization weight of l1 norm
        l2_lambd (float): regularization weight of l2 norm
        log (int): step size of evlaution and log
    """
    # create log folder
    if not os.path.isdir(logdir):
        os.makedirs(logdir, exist_ok=True)
    # use training data for test if no test data
    if testloader is None:
        testloader = trainloader
    # init tensorboard
    writer = SummaryWriter(log_dir=logdir)
    # get device
    device = getDevice()
    reg.to(device)
    # training mode
    reg.train()
    # set SPO+ Loss as criterion
    criterion = pyepo.func.SPOPlus(model, processes=processes)
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
            loss = criterion.apply(cp, c, w, z).mean()
            # add logs
            if l1_lambd or l2_lambd:
                writer.add_scalar('Train/SPO Loss', loss.item(), cnt)
            # l1 reg
            if l1_lambd:
                l1_reg = torch.abs(cp - c).mean(dim=1).mean()
                writer.add_scalar('Train/L1 Reg', l1_reg.item(), cnt)
                loss += l1_lambd * l1_reg
            # l2 reg
            if l2_lambd:
                l2_reg = ((cp - c) ** 2).mean(dim=1).mean()
                writer.add_scalar('Train/L2 Reg', l2_reg.item(), cnt)
                loss += l2_lambd * l2_reg
            # add hook
            abs_grad = []
            cp.register_hook(lambda grad:
                             abs_grad.append(torch.abs(grad).mean().item()))
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # abs grad
            writer.add_scalar('Train/Abs Grad', abs_grad[0], cnt)
            # add logs
            writer.add_scalar('Train/Total Loss', loss.item(), cnt)
            desc = "Epoch {}, Loss: {:.4f}".format(epoch, loss.item())
            pbar.set_description(desc)
            cnt += 1
        # eval
        if log and (epoch % log == 0):
            # true regret
            trueloss = pyepo.metric.regret(reg, model, testloader)
            writer.add_scalar('Eval/True SPO Loss', trueloss, epoch)
            # unambiguous regret
            # unambloss = pyepo.metric.unambRegret(reg, model, testloader)
            # writer.add_scalar('Eval/Unambiguous SPO Loss', unambloss, epoch)
    writer.close()
