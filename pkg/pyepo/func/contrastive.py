#!/usr/bin/env python
# coding: utf-8
"""
Noise contrastive estimation loss function
"""

import numpy as np
import torch

from pyepo import EPO
from pyepo.func.abcmodule import optModule
from pyepo.data.dataset import optDataset
from pyepo.func.utlis import _solveWithObj4Par, _solve_in_pass


class NCE(optModule):
    """
    An autograd module for noise contrastive estimation as surrogate loss
    functions, based on viewing suboptimal solutions as negative examples.

    For the NCE, the cost vector needs to be predicted from contextual data and
    maximizes the separation of the probability of the optimal solution.

    Thus allows us to design an algorithm based on stochastic gradient descent.

    Reference: <https://www.ijcai.org/proceedings/2021/390>
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, reduction="mean", dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (None/optDataset): the training data, usually this is simply the training set
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        # solution pool
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        # convert to tensor
        self.solpool = torch.tensor(dataset.sols.copy(), dtype=torch.float32)
        # remove duplicate
        self.solpool = torch.unique(self.solpool, dim=0)

    def forward(self, pred_cost, true_sol):
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
        # get current obj
        obj_cp = torch.einsum("bd,bd->b", pred_cost, true_sol).unsqueeze(1)
        # get obj for solpool
        objpool_cp = torch.einsum("bd,nd->bn", pred_cost, self.solpool)
        # get loss
        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss = (obj_cp - objpool_cp).mean(axis=1)
        elif self.optmodel.modelSense == EPO.MAXIMIZE:
            loss = (objpool_cp - obj_cp).mean(axis=1)
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


class contrastiveMAP(optModule):
    """
    An autograd module for Maximum A Posterior contrastive estimation as
    surrogate loss functions, which is an efficient self-contrastive algorithm.

    For the MAP, the cost vector needs to be predicted from contextual data and
    maximizes the separation of the probability of the optimal solution.

    Thus, it allows us to design an algorithm based on stochastic gradient descent.

    Reference: <https://www.ijcai.org/proceedings/2021/390>
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, reduction="mean", dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (None/optDataset): the training data, usually this is simply the training set
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        # solution pool
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        # convert to tensor
        self.solpool = torch.tensor(dataset.sols.copy(), dtype=torch.float32)
        # remove duplicate
        self.solpool = torch.unique(self.solpool, dim=0)

    def forward(self, pred_cost, true_sol):
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
        # get current obj
        obj_cp = torch.einsum("bd,bd->b", pred_cost, true_sol).unsqueeze(1)
        # get obj for solpool
        objpool_cp = torch.einsum("bd,nd->bn", pred_cost, self.solpool)
        # get loss
        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss, _ = (obj_cp - objpool_cp).max(axis=1)
        elif self.optmodel.modelSense == EPO.MAXIMIZE:
            loss, _ = (objpool_cp - obj_cp).max(axis=1)
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
