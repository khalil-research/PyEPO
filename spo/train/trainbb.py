#!/usr/bin/env python
# coding: utf-8
"""
Train with SPO+ loss
"""

import os
import time

from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

import spo
from spo.train.util import getDevice

def trainBB(reg, model, optimizer, trainloader, testloader=None,
             epoch=50, processes=1, bb_lambd=10, l1_lambd=0, l2_lambd=0, log=0):
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
        bb_lambd (float): Black-Box parameter for function smoothing 
        l1_lambd (float): regularization weight of l1 norm
        l2_lambd (float): regularization weight of l2 norm
        log (int): step size of evlaution and log
    """
    # create log folder
    if not os.path.isdir("./logs"):
        os.mkdir("./logs")
    # use training data for test if no test data
    if testloader is None:
        testloader = trainloader
    # init tensorboard
    writer = SummaryWriter(log_dir="./logs")
    # get device
    device = getDevice()
    reg.to(device)
    # training mode
    reg.train()
    # set black-box optimizer
    bb = spo.func.blackboxOpt(model, lambd=bb_lambd, processes=processes)
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
            wp = bb.apply(cp)
            # objective value
            zp = (wp * c).sum(1).view(-1, 1)
            # SPO loss
            loss = criterion(zp, z)
            # add logs
            if l1_lambd or l2_lambd:
                writer.add_scalar('Train/SPO Loss', loss.item(), cnt)
            # l1 reg
            if l1_lambd:
                l1_reg = l1_lambd * torch.abs(cp - c).sum(dim=1).mean()
                writer.add_scalar('Train/L1 Reg', l1_reg.item(), cnt)
                loss += l1_reg
            # l2 reg
            if l2_lambd:
                l2_reg = l2_lambd * ((cp - c) ** 2).sum(dim=1).mean()
                writer.add_scalar('Train/L2 Reg', l1_reg.item(), cnt)
                loss += l2_reg
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # add logs
            writer.add_scalar('Train/Total Loss', loss.item(), cnt)
            desc = "Epoch {}, Loss: {:.4f}".format(epoch, loss.item())
            pbar.set_description(desc)
            cnt += 1
        # eval
        if log and (epoch % log == 0):
            # true SPO
            trueloss = spo.eval.trueSPO(reg, model, testloader)
            writer.add_scalar('Eval/True SPO Loss', trueloss, epoch)
            # unambiguous SPO
            unambloss = spo.eval.unambSPO(reg, model, testloader)
            writer.add_scalar('Eval/Unambiguous SPO Loss', unambloss, epoch)
    writer.close()
