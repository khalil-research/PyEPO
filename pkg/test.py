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
        # create a model
        m = gp.Model()
        # variables
        x = m.addVars(self.num_item, name="x", vtype=GRB.BINARY)
        # sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(gp.quicksum([self.weights[0,i] * x[i] for i in range(self.num_item)]) <= 7)
        m.addConstr(gp.quicksum([self.weights[1,i] * x[i] for i in range(self.num_item)]) <= 8)
        m.addConstr(gp.quicksum([self.weights[2,i] * x[i] for i in range(self.num_item)]) <= 9)
        return m, x


# prediction model
class LinearRegression(nn.Module):

    def __init__(self, num_feat, num_item):
        super().__init__()
        self.linear = nn.Linear(num_feat, num_item)

    def forward(self, x):
        out = self.linear(x)
        return out


if __name__ == "__main__":

    # generate data
    num_data = 1000 # number of data
    num_feat = 5 # size of feature
    num_item = 10 # number of items
    weights, x, c = pyepo.data.knapsack.genData(num_data, num_feat, num_item, dim=3, deg=4, noise_width=0.1, seed=135)

    # init optimization model
    optmodel = myModel(weights)

    # build dataset
    dataset = pyepo.data.dataset.optDataset(optmodel, x, c)
    # get data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # training config
    num_epochs = 10

    def train_and_eval(name, loss_fn, call, lr=1e-2):
        """Train with a given loss function and print results."""
        pred = LinearRegression(num_feat, num_item)
        opt = torch.optim.Adam(pred.parameters(), lr=lr)
        print("\n--- {} ---".format(name))
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            for data in dataloader:
                x, c, w, z = data
                cp = pred(x)
                loss = call(loss_fn, cp, c, w, z)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                num_batches += 1
            avg_loss = epoch_loss / num_batches
            print("  Epoch {:3d}/{}: avg loss = {:.4f}".format(
                epoch + 1, num_epochs, avg_loss))
        reg = pyepo.metric.regret(pred, optmodel, dataloader)
        ms = pyepo.metric.MSE(pred, dataloader)
        print("  Regret: {:.4f}  MSE: {:.4f}".format(reg, ms))

    # 1. SPO+ (surrogate)
    spo = pyepo.func.SPOPlus(optmodel, processes=4)
    train_and_eval("SPOPlus", spo,
                   lambda fn, cp, c, w, z: fn(cp, c, w, z))

    # 2. Perturbation Gradient (surrogate)
    pg = pyepo.func.perturbationGradient(optmodel, processes=4, sigma=1.0)
    train_and_eval("perturbationGradient", pg,
                   lambda fn, cp, c, w, z: fn(cp, c))

    # task loss for methods that return solutions (MAXIMIZE: minimize -c^T w_hat)
    def task_loss(fn, cp, c, w, z):
        w_hat = fn(cp)
        return -(c * w_hat).sum(dim=1).mean()

    # 3. Blackbox Differentiable Optimizer
    bb = pyepo.func.blackboxOpt(optmodel, processes=4, lambd=10)
    train_and_eval("blackboxOpt", bb, task_loss)

    # 4. Negative Identity
    nid = pyepo.func.negativeIdentity(optmodel, processes=4)
    train_and_eval("negativeIdentity", nid, task_loss)

    # 5. Perturbed Optimizer
    ptb = pyepo.func.perturbedOpt(optmodel, processes=4, n_samples=5, sigma=1.0)
    train_and_eval("perturbedOpt", ptb, task_loss)

    # 6. Perturbed Fenchel-Young
    pfy = pyepo.func.perturbedFenchelYoung(optmodel, processes=4, n_samples=5, sigma=1.0)
    train_and_eval("perturbedFenchelYoung", pfy,
                   lambda fn, cp, c, w, z: fn(cp, w))

    # 7. Implicit MLE
    imle = pyepo.func.implicitMLE(optmodel, processes=4, n_samples=5, sigma=1.0)
    train_and_eval("implicitMLE", imle, task_loss)

    # 8. Adaptive Implicit MLE
    aimle = pyepo.func.adaptiveImplicitMLE(optmodel, processes=4, n_samples=5, sigma=1.0)
    train_and_eval("adaptiveImplicitMLE", aimle, task_loss)

    # 9. NCE (contrastive)
    nce = pyepo.func.NCE(optmodel, processes=1, solve_ratio=1, dataset=dataset)
    train_and_eval("NCE", nce,
                   lambda fn, cp, c, w, z: fn(cp, w))

    # 10. Contrastive MAP
    cmap = pyepo.func.contrastiveMAP(optmodel, processes=1, solve_ratio=1, dataset=dataset)
    train_and_eval("contrastiveMAP", cmap,
                   lambda fn, cp, c, w, z: fn(cp, w))

    # 11. Listwise Learning-to-Rank
    ltr = pyepo.func.listwiseLTR(optmodel, processes=1, solve_ratio=1, dataset=dataset)
    train_and_eval("listwiseLTR", ltr,
                   lambda fn, cp, c, w, z: fn(cp, c))

    # 12. Pairwise Learning-to-Rank
    pw = pyepo.func.pairwiseLTR(optmodel, processes=1, solve_ratio=1, dataset=dataset)
    train_and_eval("pairwiseLTR", pw,
                   lambda fn, cp, c, w, z: fn(cp, c))

    # 13. Pointwise Learning-to-Rank
    pt = pyepo.func.pointwiseLTR(optmodel, processes=1, solve_ratio=1, dataset=dataset)
    train_and_eval("pointwiseLTR", pt,
                   lambda fn, cp, c, w, z: fn(cp, c))
