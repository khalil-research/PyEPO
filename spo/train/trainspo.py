#!/usr/bin/env python
# coding: utf-8
"""
Train with SPO+ loss
"""

import os
import time

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import spo
from spo.train.util import getDevice

def trainSPO(trainloader, reg, model, optimizer, epoch,
             processes=1, l1_lambd=0, l2_lambd=0, log=0):
    """
    function to train PyTorch nn with SPO+ loss
    """
    # create log folder
    if not os.path.isdir("./logs"):
        os.mkdir("./logs")
    # init tensorboard
    writer = SummaryWriter(log_dir="./logs")
    # get device
    device = getDevice()
    reg.to(device)
    # training mode
    reg.train()
    # set SPO+ Loss as criterion
    criterion = spo.func.SPOPlus(model, processes=processes)
    # train
    time.sleep(1)
    pbar = tqdm(range(epoch))
    cnt = 0
    for epoch in pbar:
        # load data
        for i, data in enumerate(trainloader):
            x, c, w, z = data
            x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
            # forward pass
            cp = reg(x)
            loss = criterion.apply(cp, c, w, z).mean()
            # l1 reg
            if l1_lambd:
                l1_reg = torch.abs(cp - c).sum(dim=1).mean()
                loss += l1_lambd * l1_reg
            # l2 reg
            if l2_lambd:
                l2_reg = ((cp - c) ** 2).sum(dim=1).mean()
                loss += l2_lambd * l2_reg
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # add logs
            pbar.set_description("Epoch {}, Loss: {:.4f}". \
                                 format(epoch, loss.item()))
            writer.add_scalar('Traning Loss', loss.item(), cnt)
            cnt += 1
    writer.close()
