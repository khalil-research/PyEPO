#!/usr/bin/env python
"""
Abstract autograd optimization module
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn

from pyepo.func._common import require_solution_pool
from pyepo.func.runtime import Reduction, bind_runtime_state, init_runtime, init_solution_pool
from pyepo.func.utils import _solve_in_pass

if TYPE_CHECKING:
    import numpy as np
    from pathos.multiprocessing import ProcessingPool

    from pyepo.data.dataset import optDataset
    from pyepo.model.opt import optModel

logger = logging.getLogger(__name__)


class optModule(nn.Module):
    """
    An abstract module for differentiable optimization losses in end-to-end
    predict-then-optimize. It provides common functionality (multiprocessing,
    solution pooling, loss reduction) for all loss modules.
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
            processes: number of processors, 1 for single-core, 0 for all of cores
            solve_ratio: the ratio of new solutions computed during training
            reduction: the reduction to apply to the output
            dataset: the training data
            require_solpool: if True, always initialize solution pool from dataset
            seed: seed for the per-instance branch RNG (solve-vs-cache decision)
        """
        super().__init__()
        runtime = init_runtime(self, optmodel, processes, solve_ratio, reduction, seed, logger)
        bind_runtime_state(self, runtime)
        # framework-specific solution pool
        self.solpool = init_solution_pool(
            dataset,
            self.solve_ratio,
            require_solpool,
            lambda sols: torch.unique(sols, dim=0).clone(),
        )

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

    def _refresh_solution_pool(self, cost: torch.Tensor) -> torch.Tensor:
        """Optionally solve, then return the initialized pool on ``cost``'s device."""
        if self._branch_rng.uniform() <= self.solve_ratio:
            _, _, self.solpool = _solve_in_pass(
                cost, self.optmodel, self.processes, self.pool, self.solpool
            )
        solpool = require_solution_pool(self.solpool)
        if solpool.device != cost.device:
            solpool = solpool.to(cost.device)
            self.solpool = solpool
        return solpool
