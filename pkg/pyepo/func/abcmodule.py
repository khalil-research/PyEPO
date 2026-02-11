#!/usr/bin/env python
# coding: utf-8
"""
Abstract autograd optimization module
"""

from abc import abstractmethod
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool

import numpy as np
import torch
from torch import nn

from pyepo.data.dataset import optDataset
from pyepo.model.opt import optModel
from pyepo.model.mpax import optMpaxModel


class optModule(nn.Module):
    """
    An abstract module for differentiable optimization losses in end-to-end
    predict-then-optimize. It provides common functionality (multiprocessing,
    solution pooling, loss reduction) for all loss modules.
    """
    def __init__(self, optmodel, processes=1, solve_ratio=1, reduction="mean", dataset=None, require_solpool=False):
        """
        Args:
            optmodel (optModel): a PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (None/optDataset): the training data
            require_solpool (bool): if True, always initialize solution pool from dataset
        """
        super().__init__()
        # optimization model
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel")
        self.optmodel = optmodel
        # force processes to 1 for MPAX
        if isinstance(optmodel, optMpaxModel) and processes > 1:
            print("MPAX does not support multiprocessing. Setting `processes = 1`.")
            processes = 1
        # number of processes
        if processes < 0 or processes > mp.cpu_count():
            raise ValueError("Invalid processors number {}, only {} cores.".
                format(processes, mp.cpu_count()))
        self.processes = mp.cpu_count() if processes == 0 else processes
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
        self._solset = set()
        if self.solve_ratio < 1 or require_solpool: # init solution pool
            if not isinstance(dataset, optDataset): # type checking
                raise TypeError("dataset is not an optDataset")
            # convert to tensor and deduplicate
            sols = dataset.sols.clone()
            sols = torch.unique(sols, dim=0)
            self.solpool = sols
            # build hash set for O(1) dedup
            self._solset = {s.numpy().tobytes() for s in sols.cpu()}
        # reduction
        self.reduction = reduction

    @abstractmethod
    def forward(self, pred_cost, true_cost):
        """
        Forward pass
        """
        # convert tensor
        pass

    def _reduce(self, loss):
        """
        Apply reduction to loss tensor
        """
        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError("No reduction '{}'.".format(self.reduction))


