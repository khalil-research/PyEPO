#!/usr/bin/env python
# coding: utf-8

import numpy as np

def trueSPO(pmodel, omodel, dataloader):
    """
    evaluate model performence
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
        # predict
        cp = pmodel(x).to('cpu').detach().numpy()
        # solve
        for j in range(cp.shape[0]):
            # accumulate loss
            loss += calTrueSPO(omodel, cp[j], c[j].to('cpu').detach().numpy(), z[j].item())
        optsum += abs(z).sum().item()
    # normalized
    return loss / (optsum + 1e-3)

def calTrueSPO(omodel, pred_cost, true_cost, true_obj):
    """
    calculate normalized true SPO
    args:
      omodel: optModel
      pred_cost: predicted cost
      true_cost: true cost
      true_obj: true optimal objective value
    """
    # opt sol for pred cost
    omodel.setObj(pred_cost)
    sol, _ = omodel.solve()
    # obj with true cost
    obj = np.dot(sol, true_cost)
    # loss
    return obj - true_obj
