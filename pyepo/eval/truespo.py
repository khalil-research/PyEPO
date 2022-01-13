#!/usr/bin/env python
# coding: utf-8
"""
True SPO loss
"""

import numpy as np


def trueSPO(pmodel, omodel, dataloader):
    """
    A function to evaluate model performance with normalized true SPO

    Args:
        pmodel (nn): neural network predictor
        omodel (optModel): optimization model
        dataloader (DataLoader): Torch dataloader from optDataSet

    Returns:
        float: true SPO loss
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
        # predict
        cp = pmodel(x).to("cpu").detach().numpy()
        # solve
        for j in range(cp.shape[0]):
            # accumulate loss
            loss += calTrueSPO(omodel, cp[j], c[j].to("cpu").detach().numpy(),
                               z[j].item())
        optsum += abs(z).sum().item()
    # normalized
    return loss / (optsum + 1e-7)


def calTrueSPO(omodel, pred_cost, true_cost, true_obj):
    """
    A function to calculate normalized true SPO for a batch

    Args:
        omodel (optModel): optimization model
        pred_cost (torch.tensor): predicted costs
        true_cost (torch.tensor): true costs
        true_obj (torch.tensor): true optimal objective values

    Returns:
        float: true SPO losses
    """
    # opt sol for pred cost
    omodel.setObj(pred_cost)
    sol, _ = omodel.solve()
    # obj with true cost
    obj = np.dot(sol, true_cost)
    # loss
    return obj - true_obj
