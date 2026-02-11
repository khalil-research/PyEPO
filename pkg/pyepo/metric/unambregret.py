#!/usr/bin/env python
# coding: utf-8
"""
Unambiguous regret loss
"""

import numpy as np
import torch

from pyepo import EPO


def unambRegret(predmodel, optmodel, dataloader, tolerance=1e-5):
    """
    A function to evaluate model performance with normalized unambiguous regret

    Args:
        predmodel (nn): a regression neural network for cost prediction
        optmodel (optModel): a PyEPO optimization model
        dataloader (DataLoader): Torch dataloader from optDataSet
        tolerance (float): tolerance for optimization

    Returns:
        float: unambiguous regret loss
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
        # pred cost
        with torch.no_grad():
            cp = predmodel(x).cpu().numpy()
        # batch convert to numpy
        c_np = c.cpu().numpy()
        # solve
        for j in range(cp.shape[0]):
            # accumulate loss
            loss += calUnambRegret(optmodel, cp[j], c_np[j],
                                z[j].item(), tolerance, max_iter=10)
        optsum += abs(z).sum().item()
    # turn back train mode
    predmodel.train()
    # normalized
    return loss / (optsum + 1e-7)


def calUnambRegret(optmodel, pred_cost, true_cost, true_obj, tolerance=1e-5, max_iter=10):
    """
    A function to calculate normalized unambiguous regret for a batch

    Args:
        optmodel (optModel): optimization model
        pred_cost (np.ndarray): predicted costs
        true_cost (np.ndarray): true costs
        true_obj (float): true optimal objective value
        tolerance (float): tolerance for precision
        max_iter (int): maximum number of recursive retries

    Returns:
        float: unambiguous regret loss
    """
    if max_iter <= 0:
        raise RuntimeError("Max iterations reached in calUnambRegret.")
    # change precision
    cp = np.around(pred_cost / tolerance).astype(int)
    # opt sol for pred cost
    optmodel.setObj(cp)
    sol, objp = optmodel.solve()
    # to numpy
    if isinstance(sol, torch.Tensor):
        sol = sol.detach().cpu().numpy()
    objp = np.ceil(np.dot(cp, sol.T))
    # opt for pred cost
    if optmodel.modelSense == EPO.MINIMIZE:
        wst_optmodel = optmodel.addConstr(cp, objp+1e-2)
    elif optmodel.modelSense == EPO.MAXIMIZE:
        wst_optmodel = optmodel.addConstr(-cp, -objp+1e-2)
    else:
        raise ValueError("Invalid modelSense.")
    # opt model to find worst case
    try:
        wst_optmodel.setObj(-true_cost)
        _, obj = wst_optmodel.solve()
    except Exception:
        tolerance *= 10
        return calUnambRegret(optmodel, pred_cost, true_cost, true_obj,
                              tolerance=tolerance, max_iter=max_iter-1)
    obj = -obj
    # loss
    if optmodel.modelSense == EPO.MINIMIZE:
        loss = obj - true_obj
    elif optmodel.modelSense == EPO.MAXIMIZE:
        loss = true_obj - obj
    else:
        raise ValueError("Invalid modelSense.")
    return loss
