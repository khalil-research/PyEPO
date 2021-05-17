#!/usr/bin/env python
# coding: utf-8

import numpy as np

def trueSPO(pmodel, omodel, dataloader):
    """
    calculate normalized true SPO to evaluate model performence
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
            # opt sol for pred cost
            omodel.setObj(cp[j])
            sol, _ = omodel.solve()
            obj = np.dot(sol, c[j].to('cpu').detach().numpy())
            # accumulate loss
            loss += obj - z[j].item()
        optsum += z.sum().item()
    # normalized
    return loss / optsum
