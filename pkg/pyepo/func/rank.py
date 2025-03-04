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
from pyepo.func.utlis import _solveWithObj4Par, _solve_in_pass


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

    def __init__(self, optmodel, processes=1, solve_ratio=1, reduction="mean", dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (optDataset): the training data, usually this is simply the training set
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        # solution pool
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        # convert to tensor
        self.solpool = torch.tensor(dataset.sols.copy(), dtype=torch.float32)
        # remove duplicate
        self.solpool = torch.unique(self.solpool, dim=0)

    def forward(self, pred_cost, true_cost):
        """
        Forward pass
        """
        # get device
        device = pred_cost.device
        # to device
        self.solpool = self.solpool.to(device)
        # convert tensor
        cp = pred_cost.detach()
        # solve
        if np.random.uniform() <= self.solve_ratio:
            sol, _ = _solve_in_pass(cp, self.optmodel, self.processes, self.pool)
            # add into solpool
            self._update_solution_pool(sol)
        # obj for solpool
        objpool_c = true_cost @ self.solpool.T # true cost
        objpool_cp = pred_cost @ self.solpool.T # pred cost
        # cross entropy loss
        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss = - (F.log_softmax(objpool_cp, dim=1) * F.softmax(objpool_c, dim=1))
        elif self.optmodel.modelSense == EPO.MAXIMIZE:
            loss = - (F.log_softmax(- objpool_cp, dim=1) * F.softmax(- objpool_c, dim=1))
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        # reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(self.reduction))
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

    def __init__(self, optmodel, processes=1, solve_ratio=1, reduction="mean", dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        # solution pool
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        # function
        self.relu = nn.ReLU()
        # convert to tensor
        self.solpool = torch.tensor(dataset.sols.copy(), dtype=torch.float32)
        # remove duplicate
        self.solpool = torch.unique(self.solpool, dim=0)

    def forward(self, pred_cost, true_cost):
        """
        Forward pass
        """
        # get device
        device = pred_cost.device
        # to device
        self.solpool = self.solpool.to(device)
        # convert tensor
        cp = pred_cost.detach()
        # solve
        if np.random.uniform() <= self.solve_ratio:
            sol, _ = _solve_in_pass(cp, self.optmodel, self.processes, self.pool)
            # add into solpool
            self._update_solution_pool(sol)
        # obj for solpool
        objpool_c = torch.einsum("bd,nd->bn", true_cost, self.solpool) # true cost
        objpool_cp = torch.einsum("bd,nd->bn", pred_cost, self.solpool) # pred cost
        # best solutions for each instance
        if self.optmodel.modelSense == EPO.MINIMIZE:
            best_inds = torch.argmin(objpool_c, dim=1)  # Best solution indices
        elif self.optmodel.modelSense == EPO.MAXIMIZE:
            best_inds = torch.argmax(objpool_c, dim=1)  # Best solution indices
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        objpool_cp_best = objpool_cp.gather(1, best_inds.unsqueeze(1)).squeeze(1)
        # mask out best solution index
        batch_size, solpool_size = objpool_cp.shape
        mask = torch.ones((batch_size, solpool_size), dtype=torch.bool, device=device)
        mask.scatter_(1, best_inds.unsqueeze(1), False)
        # select the rest of the solutions
        objpool_cp_rest = objpool_cp[mask].view(batch_size, solpool_size - 1)
        # ranking loss: best v.s. rest
        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss = self.relu(objpool_cp_best.unsqueeze(1) - objpool_cp_rest).mean(dim=1)
        elif self.optmodel.modelSense == EPO.MAXIMIZE:
            loss = self.relu(objpool_cp_rest - objpool_cp_best.unsqueeze(1)).mean(dim=1)
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        # reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(self.reduction))
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

    def __init__(self, optmodel, processes=1, solve_ratio=1, reduction="mean", dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        # solution pool
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        # convert to tensor
        self.solpool = torch.tensor(dataset.sols.copy(), dtype=torch.float32)
        # remove duplicate
        self.solpool = torch.unique(self.solpool, dim=0)

    def forward(self, pred_cost, true_cost):
        """
        Forward pass
        """
        # get device
        device = pred_cost.device
        # to device
        self.solpool = self.solpool.to(device)
        # convert tensor
        cp = pred_cost.detach()
        # solve
        if np.random.uniform() <= self.solve_ratio:
            sol, _ = _solve_in_pass(cp, self.optmodel, self.processes, self.pool)
            # add into solpool
            self._update_solution_pool(sol)
        # obj for solpool as score
        objpool_c = true_cost @ self.solpool.T # true cost
        objpool_cp = pred_cost @ self.solpool.T # pred cost
        # squared loss
        loss = (objpool_c - objpool_cp).square().mean(axis=1)
        # reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(self.reduction))
        return loss
