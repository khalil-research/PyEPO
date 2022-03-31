#!/usr/bin/env python
# coding: utf-8

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pyepo
from pyepo.model.grb import optGrbModel
import torch
from torch import nn
from torch.utils.data import DataLoader


# optimization model
class myModel(optGrbModel):
    def __init__(self, weights):
        self.weights = np.array(weights)
        self.num_item = len(weights[0])
        super().__init__()

    def _getModel(self):
        # ceate a model
        m = gp.Model()
        # varibles
        x = m.addVars(self.num_item, name="x", vtype=GRB.BINARY)
        # sense (must be minimize)
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(gp.quicksum([self.weights[0,i] * x[i] for i in range(self.num_item)]) <= 7)
        m.addConstr(gp.quicksum([self.weights[1,i] * x[i] for i in range(self.num_item)]) <= 8)
        m.addConstr(gp.quicksum([self.weights[2,i] * x[i] for i in range(self.num_item)]) <= 9)
        return m, x


# prediction model
class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_feat, num_item)

    def forward(self, x):
        out = self.linear(x)
        return out


if __name__ == "__main__":

    # generate data
    num_data = 1000 # number of data
    num_feat = 5 # size of feature
    num_item = 10 # number of items
    weights, x, c = pyepo.data.knapsack.genData(num_data, num_feat, num_item, dim=3, deg=4, noise_width=0.5, seed=135)

    # init optimization model
    optmodel = myModel(weights)

    # init prediction model
    predmodel = LinearRegression()
    # set optimizer
    optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-2)
    # init SPO+ loss
    spo = pyepo.func.SPOPlus(optmodel, processes=4)

    # build dataset
    dataset = pyepo.data.dataset.optDataset(optmodel, x, c)
    # get data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # training
    num_epochs = 100
    for epoch in range(num_epochs):
        for data in dataloader:
            x, c, w, z = data
            # forward pass
            cp = predmodel(x)
            loss = spo.apply(cp, c, w, z).mean()
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # eval
    regret = pyepo.metric.regret(predmodel, optmodel, dataloader)
    print("Regret on Training Set: {:.4f}".format(regret))
