#!/usr/bin/env python
# coding: utf-8
"""
Noise Contrastive Estimation Loss function
"""

import numpy as np
import torch

from pyepo import EPO
from pyepo.func.abcmodule import optModule
from pyepo.data.dataset import optDataset
from pyepo.func.utlis import _solveWithObj4Par, _solve_in_pass, _cache_in_pass


class NCE(optModule):
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
        super().__init__(optmodel, processes, solve_ratio, dataset)
        # solution pool
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        self.solpool = np.unique(dataset.sols.copy(), axis=0) # remove duplicate

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
            # add into solpool
            self.solpool = np.concatenate((self.solpool, sol))
            # remove duplicate
            self.solpool = np.unique(self.solpool, axis=0)
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # get current obj
        obj_cp = torch.einsum("bd,bd->b", pred_cost, true_sol).unsqueeze(1)
        # get obj for solpool
        objpool_cp = torch.einsum("bd,nd->bn", pred_cost, solpool)
        # get loss
        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss = (objpool_cp - obj_cp).mean(axis=1)
        if self.optmodel.modelSense == EPO.MAXIMIZE:
            loss = (obj_cp - objpool_cp).mean(axis=1)
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
