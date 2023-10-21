#!/usr/bin/env python
# coding: utf-8
"""
Learning to rank Losses
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pyepo import EPO
from pyepo.func.abcmodule import optModule
from pyepo.data.dataset import optDataset
from pyepo.func.utlis import _solveWithObj4Par, _solve_in_pass, _cache_in_pass


class listwiseLTR(optModule):
    """
    An autograd module for listwise learning to rank, where the goal is to learn
    an objective function that ranks a pool of feasible solutions correctly.

    For the listwise LTR, the cost vector needs to be predicted from the
    contextual data and the loss measures the scores of the whole ranked lists.

    Thus, it allows us to design an algorithm based on stochastic gradient
    descent.

    Reference: <https://proceedings.mlr.press/v162/mandi22a.html>
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (optDataset): the training data, usually this is simply the training set
        """
        super().__init__(optmodel, processes, solve_ratio, dataset)
        # solution pool
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        self.solpool = np.unique(dataset.sols.copy(), axis=0) # remove duplicate

    def forward(self, pred_cost, true_cost, reduction="mean"):
        """
        Forward pass
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach().to("cpu").numpy()
        # solve
        if np.random.uniform() <= self.solve_ratio:
            sol, _ = _solve_in_pass(cp, self.optmodel, self.processes, self.pool)
            # add into solpool
            self.solpool = np.concatenate((self.solpool, sol))
            # remove duplicate
            self.solpool = np.unique(self.solpool, axis=0)
        # convert tensor
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # obj for solpool
        objpool_c = true_cost @ solpool.T # true cost
        objpool_cp = pred_cost @ solpool.T # pred cost
        # cross entropy loss
        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss = - (F.log_softmax(objpool_cp, dim=1) *
                      F.softmax(objpool_c, dim=1))
        if self.optmodel.modelSense == EPO.MAXIMIZE:
            loss = - (F.log_softmax(- objpool_cp, dim=1) *
                      F.softmax(- objpool_c, dim=1))
        # reduction
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(reduction))
        return loss


class pairwiseLTR(optModule):
    """
    An autograd module for pairwise learning to rank, where the goal is to learn
    an objective function that ranks a pool of feasible solutions correctly.

    For the pairwise LTR, the cost vector needs to be predicted from the
    contextual data and the loss learns the relative ordering of pairs of items.

    Thus, it allows us to design an algorithm based on stochastic gradient
    descent.

    Reference: <https://proceedings.mlr.press/v162/mandi22a.html>
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, dataset)
        # solution pool
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        self.solpool = np.unique(dataset.sols.copy(), axis=0) # remove duplicate

    def forward(self, pred_cost, true_cost, reduction="mean"):
        """
        Forward pass
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach().to("cpu").numpy()
        # solve
        if np.random.uniform() <= self.solve_ratio:
            sol, _ = _solve_in_pass(cp, self.optmodel, self.processes, self.pool)
            # add into solpool
            self.solpool = np.concatenate((self.solpool, sol))
            # remove duplicate
            self.solpool = np.unique(self.solpool, axis=0)
        # convert tensor
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # obj for solpool
        objpool_c = torch.einsum("bd,nd->bn", true_cost, solpool) # true cost
        objpool_cp = torch.einsum("bd,nd->bn", pred_cost, solpool) # pred cost
        # init relu as max(0,x)
        relu = nn.ReLU()
        # init loss
        loss = []
        for i in range(len(pred_cost)):
            # best sol
            if self.optmodel.modelSense == EPO.MINIMIZE:
                best_ind = torch.argmin(objpool_c[i])
            if self.optmodel.modelSense == EPO.MAXIMIZE:
                best_ind = torch.argmax(objpool_c[i])
            objpool_cp_best = objpool_cp[i, best_ind]
            # rest sol
            rest_ind = [j for j in range(len(objpool_cp[i])) if j != best_ind]
            objpool_cp_rest = objpool_cp[i, rest_ind]
            # best vs rest loss
            if self.optmodel.modelSense == EPO.MINIMIZE:
                loss.append(relu(objpool_cp_best - objpool_cp_rest).mean())
            if self.optmodel.modelSense == EPO.MAXIMIZE:
                loss.append(relu(objpool_cp_rest - objpool_cp_best).mean())
        loss = torch.stack(loss)
        # reduction
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(reduction))
        return loss


class pointwiseLTR(optModule):
    """
    An autograd module for pointwise learning to rank, where the goal is to
    learn an objective function that ranks a pool of feasible solutions
    correctly.

    For the pointwise LTR, the cost vector needs to be predicted from contextual
    data, and calculates the ranking scores of the items.

    Thus, it allows us to design an algorithm based on stochastic gradient
    descent.

    Reference: <https://proceedings.mlr.press/v162/mandi22a.html>
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, dataset)
        # solution pool
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        self.solpool = np.unique(dataset.sols.copy(), axis=0) # remove duplicate

    def forward(self, pred_cost, true_cost, reduction="mean"):
        """
        Forward pass
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach().to("cpu").numpy()
        # solve
        if np.random.uniform() <= self.solve_ratio:
            sol, _ = _solve_in_pass(cp, self.optmodel, self.processes, self.pool)
            # add into solpool
            self.solpool = np.concatenate((self.solpool, sol))
            # remove duplicate
            self.solpool = np.unique(self.solpool, axis=0)
        # convert tensor
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # obj for solpool as score
        objpool_c = true_cost @ solpool.T # true cost
        objpool_cp = pred_cost @ solpool.T # pred cost
        # squared loss
        loss = (objpool_c - objpool_cp).square().mean(axis=1)
        # reduction
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(reduction))
        return loss
