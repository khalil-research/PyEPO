#!/usr/bin/env python
# coding: utf-8
"""
Model evaluation
"""

from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import time

import pyepo

def eval(trainset, testset, predmodel, optmodel, config):
    """
    evaluate performance
    """
    print("Evaluating...")
    if config.mthd == "2s":
        # prediction
        c_test_pred = predmodel.predict(testset.feats)
        truespo = 0
        unambspo = 0
        mse = 0
        for i in tqdm(range(len(testset))):
            cp_i = c_test_pred[i]
            c_i = testset.costs[i]
            z_i = testset.objs[i,0]
            truespo += pyepo.metric.calRegret(optmodel, cp_i, c_i, z_i)
            unambspo += pyepo.metric.calUnambRegret(optmodel, cp_i, c_i, z_i)
        mse = ((c_test_pred - testset.costs) ** 2).mean()
        truespo /= abs(testset.objs.sum() + 1e-3)
        unambspo /= abs(testset.objs.sum() + 1e-3)
        time.sleep(1)
    else:
        testloader = DataLoader(testset, batch_size=config.batch, shuffle=False)
        # DFJ is fastest for unambSPO
        if config.prob == "tsp":
            optmodel = pyepo.model.grb.tspDFJModel(config.nodes)
        if config.scal:
            # rescale multiplier
            scal = getScal(predmodel, trainset)
            # new model
            predmodel = rescalePredModel(predmodel, scal)
        truespo = pyepo.metric.regret(predmodel, optmodel, testloader)
        unambspo = pyepo.metric.unambRegret(predmodel, optmodel, testloader)
        mse = pyepo.metric.MSE(predmodel, testloader)
    print('Normalized true SPO Loss: {:.2f}%'.format(truespo * 100))
    print('Normalized unambiguous SPO Loss: {:.2f}%'.format(unambspo * 100))
    print('MSE Loss: {:.2f}'.format(mse))
    return truespo, unambspo, mse


def getScal(predmodel, dataset):
    """
    get rescale multiplier
    """
    # evaluate mode
    predmodel.eval()
    # init list of ratio
    r_list = []
    # load data
    for data in dataset:
        x, c, w, z = data
        # cuda
        if next(predmodel.parameters()).is_cuda:
            x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
        # predict
        cp = predmodel(x)
        # ratio
        r = c.to("cpu").detach().numpy() / cp.to("cpu").detach().numpy()
        # remove negaitive
        r = r[r >= 1e-5]
        r_list.append(r.mean())
    return np.mean(r_list)



def rescalePredModel(predmodel, scal):
    """
    get rescale model
    """
    from torch import nn
    class rescaleFcNet(nn.Module):
        """
        multi-layer fully connected neural network regression
        """
        def __init__(self, predmodel, scal):
            super().__init__()
            self.scal = scal
            self.main = predmodel

        def forward(self, x):
            x = self.main(x)
            return self.scal * x
    return rescaleFcNet(predmodel, scal)
