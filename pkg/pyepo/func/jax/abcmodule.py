#!/usr/bin/env python
"""
Abstract optimization module
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpy as np

from pyepo import EPO
from pyepo.model.opt import optModel


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
            processes: number of solver processes (1 = single-core)
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
        # multiprocessing not yet supported in the JAX frontend
        if processes != 1:
            raise NotImplementedError(
                f"processes={processes} not supported in pyepo.func.jax yet (single-core only)"
            )
        self.processes = processes
        # solution-pool caching: the branch RNG decides solve-vs-cache each pass
        if not (0 <= solve_ratio <= 1):
            raise ValueError(f"Invalid solving ratio {solve_ratio}. It should be between 0 and 1.")
        self.solve_ratio = solve_ratio
        self._branch_rng = np.random.RandomState(seed)
        # solution pool: seeded when caching or a pool loss needs it; grows in solve_or_cache
        self.solpool = None
        if self.solve_ratio < 1 or require_solpool:
            from pyepo.data.dataset import optDataset

            if not isinstance(dataset, optDataset):
                raise TypeError("dataset is not an optDataset")
            self.solpool = jnp.unique(jnp.asarray(dataset.sols), axis=0)
        # reduction
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"No reduction '{reduction}'.")
        self.reduction = reduction

    def __call__(self, *args):
        return self.forward(*args)

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
