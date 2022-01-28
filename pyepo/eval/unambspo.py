#!/usr/bin/env python
# coding: utf-8
"""
Unambiguous SPO loss
"""

import copy

import numpy as np

from pyepo import EPO


def unambSPO(predmodel, optmodel, dataloader, tolerance=1e-5):
    """
    A function to evaluate model performance with normalized unambiguous SPO

    Args:
        predmodel (nn): neural network predictor
        optmodel (optModel): optimization model
        dataloader (DataLoader): Torch dataloader from optDataSet

    Returns:
        float: unambiguous SPO loss
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
        # pred cost
        cp = predmodel(x).to("cpu").detach().numpy()
        # solve
        for j in range(cp.shape[0]):
            # accumulate loss
            loss += calUnambSPO(optmodel, cp[j], c[j].to("cpu").detach().numpy(),
                                z[j].item(), tolerance)
        optsum += abs(z).sum().item()
    # normalized
    return loss / (optsum + 1e-7)


def calUnambSPO(optmodel, pred_cost, true_cost, true_obj, tolerance=1e-5):
    """
    A function to calculate normalized unambiguous SPO for a batch

    Args:
        optmodel (optModel): optimization model
        pred_cost (torch.tensor): predicted costs
        true_cost (torch.tensor): true costs
        true_obj (torch.tensor): true optimal objective values

    Returns:
        float: unambiguous SPO losses
    """
    # change precision
    cp = np.around(pred_cost / tolerance).astype(int)
    # opt sol for pred cost
    optmodel.setObj(cp)
    sol, objp = optmodel.solve()
    sol = np.array(sol)
    objp = np.ceil(np.dot(cp, sol.T))
    # opt for pred cost
    if optmodel.modelSense == EPO.MINIMIZE:
        wst_optmodel = optmodel.addConstr(cp, objp+1e-2)
    if optmodel.modelSense == EPO.MAXIMIZE:
        wst_optmodel = optmodel.addConstr(-cp, -objp+1e-2)
    # opt model to find worst case
    try:
        wst_optmodel.setObj(-true_cost)
        _, obj = wst_optmodel.solve()
    except:
        tolerance *= 10
        return calUnambSPO(optmodel, pred_cost, true_cost, true_obj, tolerance=tolerance)
    obj = -obj
    # loss
    if optmodel.modelSense == EPO.MINIMIZE:
        loss = obj - true_obj
    if optmodel.modelSense == EPO.MAXIMIZE:
        loss = true_obj - obj
    return loss
