#!/usr/bin/env python
# coding: utf-8
"""
Mean Squared Error
"""

import torch


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
    # get device
    device = next(predmodel.parameters()).device
    # load data
    with torch.no_grad():
        for data in dataloader:
            x, c, w, z = data
            x, c = x.to(device), c.to(device)
            # predict
            cp = predmodel(x)
            loss += ((cp - c) ** 2).mean(dim=1).sum().item()
    # restore training mode
    predmodel.train()
    return loss / len(dataloader.dataset)
