#!/usr/bin/env python
# coding: utf-8
"""
Unambiguous SPO loss
"""

import copy

import numpy as np

from pyepo import EPO


def unambSPO(pmodel, omodel, dataloader, tolerance=1e-5):
    """
    A function to evaluate model performance with normalized unambiguous SPO

    Args:
        pmodel (nn): neural network predictor
        omodel (optModel): optimization model
        dataloader (DataLoader): Torch dataloader from optDataSet

    Returns:
        float: unambiguous SPO loss
    """
    # evaluate
    pmodel.eval()
    loss = 0
    optsum = 0
    # load data
    for data in dataloader:
        x, c, w, z = data
        # cuda
        if next(pmodel.parameters()).is_cuda:
            x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
        # pred cost
        cp = pmodel(x).to("cpu").detach().numpy()
        # solve
        for j in range(cp.shape[0]):
            # accumulate loss
            loss += calUnambSPO(omodel, cp[j], c[j].to("cpu").detach().numpy(),
                                z[j].item(), tolerance)
        optsum += abs(z).sum().item()
    # normalized
    return loss / (optsum + 1e-7)


def calUnambSPO(omodel, pred_cost, true_cost, true_obj, tolerance=1e-5):
    """
    A function to calculate normalized unambiguous SPO for a batch

    Args:
        omodel (optModel): optimization model
        pred_cost (torch.tensor): predicted costs
        true_cost (torch.tensor): true costs
        true_obj (torch.tensor): true optimal objective values

    Returns:
        float: unambiguous SPO losses
    """
    # change precision
    cp = np.around(pred_cost / tolerance).astype(int)
    # opt sol for pred cost
    omodel.setObj(cp)
    sol, objp = omodel.solve()
    sol = np.array(sol)
    objp = np.ceil(np.dot(cp, sol.T))
    # opt for pred cost
    if omodel.modelSense == EPO.MINIMIZE:
        wst_omodel = omodel.addConstr(cp, objp+1e-2)
    if omodel.modelSense == EPO.MAXIMIZE:
        wst_omodel = omodel.addConstr(-cp, -objp+1e-2)
    # opt model to find worst case
    try:
        wst_omodel.setObj(-true_cost)
        _, obj = wst_omodel.solve()
    except:
        tolerance *= 10
        return calUnambSPO(omodel, pred_cost, true_cost, true_obj, tolerance=tolerance)
    obj = -obj
    # loss
    if omodel.modelSense == EPO.MINIMIZE:
        loss = obj - true_obj
    if omodel.modelSense == EPO.MAXIMIZE:
        loss = true_obj - obj
    return loss
