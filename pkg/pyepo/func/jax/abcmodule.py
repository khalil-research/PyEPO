#!/usr/bin/env python
"""
Abstract optimization module
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import jax.numpy as jnp

from pyepo.func.runtime import init_runtime

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
        runtime = init_runtime(self, optmodel, processes, solve_ratio, reduction, seed, logger)
        self.optmodel = runtime.optmodel
        self.processes = runtime.processes
        self.pool = runtime.pool
        self.solve_ratio = runtime.solve_ratio
        self.reduction = runtime.reduction
        self._branch_rng = runtime.branch_rng
        # framework-specific solution pool
        self.solpool = None
        if self.solve_ratio < 1 or require_solpool:  # init solution pool
            from pyepo.data.dataset import optDataset

            if not isinstance(dataset, optDataset):  # type checking
                raise TypeError("dataset is not an optDataset")
            self.solpool = jnp.unique(jnp.asarray(dataset.sols), axis=0)

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
