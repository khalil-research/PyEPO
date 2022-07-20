#!/usr/bin/env python
# coding: utf-8
"""
True regret loss
"""

import numpy as np

from pyepo import EPO

def MSE(predmodel, dataloader):
    """
    A function to evaluate model performance with MSE

    Args:
        predmodel (nn): a regression neural network for cost prediction
        dataloader (DataLoader): Torch dataloader from optDataSet

    Returns:
        float: MSE loss
    """
    # evaluate
    predmodel.eval()
    loss = 0
    # load data
    for data in dataloader:
        x, c, w, z = data
        # cuda
        if next(predmodel.parameters()).is_cuda:
            x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
        # predict
        cp = predmodel(x)
        loss += ((cp - c) ** 2).mean(dim=1).sum().detach().data.to("cpu").numpy()
    return loss / len(dataloader.dataset)
