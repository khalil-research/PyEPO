#!/usr/bin/env python
"""
Abstract autograd optimization module
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import weakref
from abc import abstractmethod
from typing import Literal

import numpy as np
import torch
from pathos.multiprocessing import ProcessingPool
from torch import nn

from pyepo import EPO
from pyepo.data.dataset import optDataset
from pyepo.func.utils import _close_pool, _init_worker_model
from pyepo.model.mpax import optMpaxModel
from pyepo.model.opt import optModel

logger = logging.getLogger(__name__)

Reduction = Literal["mean", "sum", "none"]


class optModule(nn.Module):
    """
    An abstract module for differentiable optimization losses in end-to-end
    predict-then-optimize. It provides common functionality (multiprocessing,
    solution pooling, loss reduction) for all loss modules.
    """

    def __init__(
        self,
        optmodel: optModel,
        processes: int = 1,
        solve_ratio: float = 1.0,
        reduction: Reduction = "mean",
        dataset: optDataset | None = None,
        require_solpool: bool = False,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            processes: number of processors, 1 for single-core, 0 for all of cores
            solve_ratio: the ratio of new solutions computed during training
            reduction: the reduction to apply to the output
            dataset: the training data
            require_solpool: if True, always initialize solution pool from dataset
            seed: seed for the per-instance branch RNG (solve-vs-cache decision)
        """
        super().__init__()
        # optimization model
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel")
        # objective sense
        if optmodel.modelSense not in (EPO.MINIMIZE, EPO.MAXIMIZE):
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        self.optmodel = optmodel
        # force processes to 1 for MPAX
        if isinstance(optmodel, optMpaxModel) and processes > 1:
            logger.warning("MPAX does not support multiprocessing. Setting `processes = 1`.")
            processes = 1
        # number of processes
        if processes < 0 or processes > mp.cpu_count():
            raise ValueError(f"Invalid processors number {processes}, only {mp.cpu_count()} cores.")
        self.processes = mp.cpu_count() if processes == 0 else processes
        # single-core
        if self.processes == 1:
            self.pool = None
        # multi-core: each worker builds its own optmodel once via the initializer
        else:
            self.pool = ProcessingPool(
                self.processes,
                initializer=_init_worker_model,
                initargs=(optmodel.to_spec(),),
            )
            # release worker processes when this module is garbage-collected
            weakref.finalize(self, _close_pool, self.pool)
        logger.info("Num of cores: %d", self.processes)
        # solution pool
        self.solve_ratio = solve_ratio
        if (self.solve_ratio < 0) or (self.solve_ratio > 1):
            raise ValueError(
                f"Invalid solving ratio {self.solve_ratio}. It should be between 0 and 1."
            )
        self.solpool = None
        if self.solve_ratio < 1 or require_solpool:  # init solution pool
            if not isinstance(dataset, optDataset):  # type checking
                raise TypeError("dataset is not an optDataset")
            # dedup on dataset.sols' device
            self.solpool = torch.unique(dataset.sols, dim=0).clone()
        # per-instance RNG for the solve-vs-cache branch
        self._branch_rng = np.random.RandomState(seed)
        # reduction
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"No reduction '{reduction}'.")
        self.reduction = reduction

    @abstractmethod
    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """

    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Apply reduction to loss tensor
        """
        if self.reduction == "mean":
            return torch.mean(loss)
        if self.reduction == "sum":
            return torch.sum(loss)
        return loss
