#!/usr/bin/env python
# coding: utf-8
"""
True regret loss
"""

import numpy as np
import torch

from pyepo import EPO

def regret(predmodel, optmodel, dataloader):
    """
    A function to evaluate model performance with normalized true regret

    Args:
        predmodel (nn): a regression neural network for cost prediction
        optmodel (optModel): a PyEPO optimization model
        dataloader (DataLoader): Torch dataloader from optDataSet

    Returns:
        float: true regret loss
    """
    # evaluate
    predmodel.eval()
    loss = 0
    optsum = 0
    # get device
    device = next(predmodel.parameters()).device
    # load data
    for data in dataloader:
        x, c, w, z = data
        x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
        # predict
        with torch.no_grad():
            cp = predmodel(x).cpu().numpy()
        # batch convert to numpy
        c_np = c.cpu().numpy()
        # solve
        for j in range(cp.shape[0]):
            # accumulate loss
            loss += calRegret(optmodel, cp[j], c_np[j], z[j].item())
        optsum += abs(z).sum().item()
    # turn back train mode
    predmodel.train()
    # normalized
    return loss / (optsum + 1e-7)


def calRegret(optmodel, pred_cost, true_cost, true_obj):
    """
    A function to calculate normalized true regret for a batch

    Args:
        optmodel (optModel): optimization model
        pred_cost (np.ndarray): predicted costs
        true_cost (np.ndarray): true costs
        true_obj (float): true optimal objective value

    Returns:
        float: true regret loss
    """
    # opt sol for pred cost
    optmodel.setObj(pred_cost)
    sol, _ = optmodel.solve()
    # to numpy
    if isinstance(sol, torch.Tensor):
        sol = sol.detach().cpu().numpy()
    # obj with true cost
    obj = np.dot(sol, true_cost)
    # loss
    if optmodel.modelSense == EPO.MINIMIZE:
        loss = obj - true_obj
    elif optmodel.modelSense == EPO.MAXIMIZE:
        loss = true_obj - obj
    else:
        raise ValueError("Invalid modelSense.")
    return loss
