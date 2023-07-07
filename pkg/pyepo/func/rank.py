#!/usr/bin/env python
# coding: utf-8
"""
Learning To Rank Loss functions
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
        An autograd module for the listwise learning to rank loss.
        For the listwise learning to rank loss, the constraints are known and fixed,
        but the cost vector needs to be predicted from contextual data.
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
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # get obj for solpool
        objpool_c = true_cost @ solpool.T
        objpool_cp = pred_cost @ solpool.T
        # get cross entropy loss
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
        An autograd module for the pairwise learning to rank loss.
        For the pairwise learning to rank loss, the constraints are known and fixed,
        but the cost vector needs to be predicted from contextual data.
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
        # get obj for solpool
        objpool_c = torch.einsum("bd,nd->bn", true_cost, solpool)
        objpool_cp = torch.einsum("bd,nd->bn", pred_cost, solpool)
        # init relu as max(0,x)
        relu = nn.ReLU()
        # get loss
        loss = []
        for i in range(len(pred_cost)):
            _, indices = np.unique(objpool_c[i].cpu().detach().numpy(), return_index=True)
            if self.optmodel.modelSense == EPO.MINIMIZE:
                indices = indices[::-1].copy()
            best_ind, rest_ind = indices[0], indices[1:]
            if self.optmodel.modelSense == EPO.MINIMIZE:
                loss.append(relu(objpool_cp[i,rest_ind] - objpool_cp[i,best_ind]).mean())
            if self.optmodel.modelSense == EPO.MAXIMIZE:
                loss.append(relu(objpool_cp[i,best_ind] - objpool_cp[i,rest_ind]).mean())
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
        An autograd module for the pointwise learning to rank loss.
        For the pointwise learning to rank loss, the constraints are known and fixed,
        but the cost vector needs to be predicted from contextual data.
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
        # get obj for solpool as score
        objpool_c = true_cost @ solpool.T
        objpool_cp = pred_cost @ solpool.T
        # get squared loss
        loss = (objpool_c - objpool_cp).square()
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
