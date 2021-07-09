#!/usr/bin/env python
# coding: utf-8

import copy
import numpy as np

def unambSPO(pmodel, omodel, dataloader, tolerance=1e-4):
    """
    calculate normalized unambiguous SPO to evaluate model performence
    args:
      pmodel: prediction model
      omodel: optModel
      dataloader: dataloader from optDataSet
    """
    # evaluate
    pmodel.eval()
    loss = 0
    optsum = 0
    # load data
    for i, data in enumerate(dataloader):
        x, c, w, z = data
        # cuda
        if next(pmodel.parameters()).is_cuda:
            x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
        # pred cost
        cp = pmodel(x).to('cpu').detach().numpy()
        # solve
        for j in range(cp.shape[0]):
            # accumulate loss
            loss += calUnambSPO(omodel, cp[j], c[j].to('cpu').detach().numpy(), z[j].item(), tolerance)
        optsum += abs(z).sum().item()
    # normalized
    return loss / (optsum + 1e-3)

def calUnambSPO(omodel, pred_cost, true_cost, true_obj, tolerance=1e-4):
    """
    calculate normalized unambiguous SPO
    args:
      omodel: optModel
      pred_cost: predicted cost
      true_cost: true cost
      true_obj: true optimal objective value
    """
    # change precision
    pred_cost = np.around(pred_cost / tolerance).astype(int)
    # opt sol for pred cost
    omodel.setObj(pred_cost)
    sol, objp = omodel.solve()
    sol = np.array(sol)
    objp = np.ceil(np.dot(pred_cost, sol.T))
    # opt for pred cost
    wst_omodel = omodel.addConstr(pred_cost, objp)
    # opt model to find worst case
    wst_omodel.setObj(-true_cost)
    # solve
    _, obj = wst_omodel.solve()
    obj = -obj
    # loss
    return obj - true_obj
