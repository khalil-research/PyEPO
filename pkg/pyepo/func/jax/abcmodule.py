#!/usr/bin/env python
"""
Abstract optimization module
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import weakref
from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpy as np
from pathos.multiprocessing import ProcessingPool

from pyepo import EPO
from pyepo.func.utils import _close_pool, _init_worker_model
from pyepo.model.mpax import optMpaxModel
from pyepo.model.opt import optModel

logger = logging.getLogger(__name__)


class optModule(ABC):
    """
    An abstract module for differentiable optimization losses in end-to-end
    predict-then-optimize. It provides shared init validation, loss reduction,
    and the solution pool for all loss modules.
    """

    def __init__(
        self,
        optmodel,
        processes=1,
        solve_ratio=1.0,
        reduction="mean",
        dataset=None,
        require_solpool=False,
        seed=None,
    ):
        """
        Args:
            optmodel: a PyEPO optimization model
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of instances solved exactly each step (< 1 enables caching)
            reduction: reduction applied to the batch loss ("mean", "sum", "none")
            dataset: training dataset used to seed the solution pool when solve_ratio < 1
            require_solpool: always seed the solution pool from dataset
            seed: seed for the per-pass solve-vs-cache branch RNG
        """
        # optimization model
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel")
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
            from pyepo.data.dataset import optDataset

            if not isinstance(dataset, optDataset):  # type checking
                raise TypeError("dataset is not an optDataset")
            self.solpool = jnp.unique(jnp.asarray(dataset.sols), axis=0)
        # per-instance RNG for the solve-vs-cache branch
        self._branch_rng = np.random.RandomState(seed)
        # reduction
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"No reduction '{reduction}'.")
        self.reduction = reduction

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args):
        """
        Forward pass
        """

    def _reduce(self, loss):
        """
        Apply reduction to loss
        """
        if self.reduction == "mean":
            return jnp.mean(loss)
        if self.reduction == "sum":
            return jnp.sum(loss)
        # "none" — guaranteed valid by __init__
        return loss
