#!/usr/bin/env python
# coding: utf-8

import numpy as np

def SPOEval(pmodel, omodel, dataloader):
    """
    calculate normalized SPO to evaluate model performence
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
        cp = pmodel(x)
        # solve
        for j in range(cp.shape[0]):
            omodel.setObj(cp[j].to('cpu').detach().numpy())
            sol, _ = omodel.solve()
            obj = np.dot(sol, c[j].to('cpu').detach().numpy())
            # accumulate loss
            loss += obj - z[j].item()
        optsum += z.sum().item()
    # normalized
    return loss / optsum
