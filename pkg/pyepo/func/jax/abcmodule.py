#!/usr/bin/env python
"""
Abstract optimization module
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax.numpy as jnp

from pyepo.func._common import require_solution_pool
from pyepo.func.runtime import Reduction, bind_runtime_state, init_runtime, init_solution_pool

if TYPE_CHECKING:
    import numpy as np
    from pathos.multiprocessing import ProcessingPool

    from pyepo.data.dataset import optDataset
    from pyepo.model.opt import optModel

logger = logging.getLogger(__name__)


class optModule(ABC):
    """
    An abstract module for differentiable optimization losses in end-to-end
    predict-then-optimize. It provides shared init validation, loss reduction,
    and the solution pool for all loss modules.
    """

    optmodel: optModel
    processes: int
    pool: ProcessingPool | None
    solve_ratio: float
    reduction: Reduction
    _branch_rng: np.random.RandomState

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
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of instances solved exactly each step (< 1 enables caching)
            reduction: reduction applied to the batch loss ("mean", "sum", "none")
            dataset: training dataset used to seed the solution pool when solve_ratio < 1
            require_solpool: always seed the solution pool from dataset
            seed: seed for the per-pass solve-vs-cache branch RNG
        """
        runtime = init_runtime(self, optmodel, processes, solve_ratio, reduction, seed, logger)
        bind_runtime_state(self, runtime)
        # framework-specific solution pool
        self.solpool = init_solution_pool(
            dataset,
            self.solve_ratio,
            require_solpool,
            lambda sols: jnp.unique(jnp.asarray(sols), axis=0),
        )

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

    def _refresh_solution_pool(self, cost):
        """Optionally solve, then return the initialized JAX solution pool."""
        from pyepo.func.jax.utils import _grow_solpool

        _grow_solpool(self, cost)
        return require_solution_pool(self.solpool)
