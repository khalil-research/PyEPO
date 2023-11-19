#!/usr/bin/env python
# coding: utf-8
"""
Abstract autograd optimization module
"""

from abc import abstractmethod
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool

import numpy as np
from torch import nn

from pyepo.data.dataset import optDataset
from pyepo.model.opt import optModel


class optModule(nn.Module):
    """
        An abstract module for the learning to rank losses, which measure the difference in how the predicted cost
        vector and the true cost vector rank a pool of feasible solutions.
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
        if self.processes == 1:
            self.pool = None
        # multi-core
        else:
            self.pool = ProcessingPool(self.processes)
        print("Num of cores: {}".format(self.processes))
        # solution pool
        self.solve_ratio = solve_ratio
        if (self.solve_ratio < 0) or (self.solve_ratio > 1):
            raise ValueError("Invalid solving ratio {}. It should be between 0 and 1.".
                format(self.solve_ratio))
        self.solpool = None
        if self.solve_ratio < 1: # init solution pool
            if not isinstance(dataset, optDataset): # type checking
                raise TypeError("dataset is not an optDataset")
            self.solpool = np.unique(dataset.sols.copy(), axis=0) # remove duplicate

    @abstractmethod
    def forward(self, pred_cost, true_cost, reduction="mean"):
        """
        Forward pass
        """
        # convert tensor
        pass
