#!/usr/bin/env python
# coding: utf-8
"""
True SPO loss
"""

import numpy as np

from pyepo import EPO

def trueSPO(predmodel, optmodel, dataloader):
    """
    A function to evaluate model performance with normalized true SPO

    Args:
        predmodel (nn): neural network predictor
        optmodel (optModel): optimization model
        dataloader (DataLoader): Torch dataloader from optDataSet

    Returns:
        float: true SPO loss
    """
    # evaluate
    predmodel.eval()
    loss = 0
    optsum = 0
    # load data
    for data in dataloader:
        x, c, w, z = data
        # cuda
        if next(predmodel.parameters()).is_cuda:
            x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
        # predict
        cp = predmodel(x).to("cpu").detach().numpy()
        # solve
        for j in range(cp.shape[0]):
            # accumulate loss
            loss += calTrueSPO(optmodel, cp[j], c[j].to("cpu").detach().numpy(),
                               z[j].item())
        optsum += abs(z).sum().item()
    # normalized
    return loss / (optsum + 1e-7)


def calTrueSPO(optmodel, pred_cost, true_cost, true_obj):
    """
    A function to calculate normalized true SPO for a batch

    Args:
        optmodel (optModel): optimization model
        pred_cost (torch.tensor): predicted costs
        true_cost (torch.tensor): true costs
        true_obj (torch.tensor): true optimal objective values

    Returns:predmodel
        float: true SPO losses
    """
    # opt sol for pred cost
    optmodel.setObj(pred_cost)
    sol, _ = optmodel.solve()
    # obj with true cost
    obj = np.dot(sol, true_cost)
    # loss
    if optmodel.modelSense == EPO.MINIMIZE:
        loss = obj - true_obj
    if optmodel.modelSense == EPO.MAXIMIZE:
        loss = true_obj - obj
    return loss
