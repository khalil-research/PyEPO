#!/usr/bin/env python
# coding: utf-8
"""
Noise Contrastive Estimation Loss function
"""

import multiprocessing as mp

import numpy as np
import torch
from pathos.multiprocessing import ProcessingPool
from torch import nn

from pyepo.data.dataset import optDataset
from pyepo.model.opt import optModel

from pyepo.func.utlis import _solveWithObj4Par, _solve_in_pass, _cache_in_pass


class NCE(nn.Module):
    """
        An autograd module for the noise contrastive estimation loss.
        For the noise contrastive loss, the constraints are known and fixed,
        but the cost vector needs to be predicted from contextual data.
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__()
        # optimization model
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel")
        self.optmodel = optmodel
        # number of processes
        if processes not in range(mp.cpu_count()+1):
            raise ValueError("Invalid processors number {}, only {} cores.".
                format(processes, mp.cpu_count()))
        self.processes = mp.cpu_count() if not processes else processes
        # single-core
        if processes == 1:
            self.pool = None
        # multi-core
        else:
            self.pool = ProcessingPool(processes)
        print("Num of cores: {}".format(self.processes))
        # solution pool
        self.solve_ratio = solve_ratio
        if (self.solve_ratio < 0) or (self.solve_ratio > 1):
            raise ValueError("Invalid solving ratio {}. It should be between 0 and 1.".
                format(self.solve_ratio))
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        self.solpool = dataset.sols.copy()

    def forward(self, pred_cost, true_sol, reduction="mean"):
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
            self.solpool = np.concatenate((self.solpool, sol))
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # get loss
        loss = []
        for i in range(len(pred_cost)):
            obj_cp_i = torch.matmul(pred_cost[i], true_sol[i])
            solpool_obj_cp_i = torch.matmul(pred_cost[i], solpool.T)
            loss.append(self.optmodel.modelSense * (obj_cp_i - solpool_obj_cp_i).sum())
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
