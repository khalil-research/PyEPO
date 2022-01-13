#!/usr/bin/env python
# coding: utf-8
"""
Model evaluation
"""

from tqdm import tqdm
from torch.utils.data import DataLoader
import time

import pyepo

def eval(testset, res, model, config):
    """
    evaluate permance
    """
    print("Evaluating...")
    if config.mthd == "2s":
        # prediction
        c_test_pred = res.predict(testset.feats)
        truespo = 0
        unambspo = 0
        for i in tqdm(range(len(testset))):
            cp_i = c_test_pred[i]
            c_i = testset.costs[i]
            z_i = testset.objs[i,0]
            truespo += pyepo.eval.calTrueSPO(model, cp_i, c_i, z_i)
            unambspo += pyepo.eval.calUnambSPO(model, cp_i, c_i, z_i)
        truespo /= abs(testset.objs.sum() + 1e-3)
        unambspo /= abs(testset.objs.sum() + 1e-3)
        time.sleep(1)
    if (config.mthd == "spo") or (config.mthd == "bb"):
        testloader = DataLoader(testset, batch_size=config.batch, shuffle=False)
        # DFJ is fastest for unambSPO
        if config.prob == "tsp":
            model = pyepo.model.grb.tspDFJModel(config.nodes)
        truespo = pyepo.eval.trueSPO(res, model, testloader)
        unambspo = pyepo.eval.unambSPO(res, model, testloader)
    print('Normalized true SPO Loss: {:.2f}%'.format(truespo * 100))
    print('Normalized unambiguous SPO Loss: {:.2f}%'.format(unambspo * 100))
    return truespo, unambspo
