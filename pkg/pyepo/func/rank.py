#!/usr/bin/env python
"""
Learning to rank Losses
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from pyepo import EPO
from pyepo.func.abcmodule import optModule
from pyepo.func.utils import _solve_in_pass

if TYPE_CHECKING:
    from pyepo.data.dataset import optDataset
    from pyepo.func.abcmodule import Reduction
    from pyepo.model.opt import optModel


class listwiseLTR(optModule):
    """
    An autograd module for listwise learning to rank, where the goal is to learn
    an objective function that ranks a pool of feasible solutions correctly.

    For the listwise LTR, the cost vector needs to be predicted from the
    contextual data and the loss measures the scores of the whole ranked lists.

    Thus, it allows us to design an algorithm based on stochastic gradient
    descent.

    Reference: <https://proceedings.mlr.press/v162/mandi22a.html>
    """

    def __init__(
        self,
        optmodel: optModel,
        processes: int = 1,
        solve_ratio: float = 1.0,
        reduction: Reduction = "mean",
        dataset: optDataset | None = None,
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            processes: number of processors, 1 for single-core, 0 for all of cores
            solve_ratio: the ratio of new solutions computed during training
            reduction: the reduction to apply to the output
            dataset: the training data, usually this is simply the training set
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset, require_solpool=True)

    def forward(self, pred_cost: torch.Tensor, true_cost: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # convert tensor
        cp = pred_cost.detach()
        # solve and update pool
        if self._branch_rng.uniform() <= self.solve_ratio:
            _, _, self.solpool = _solve_in_pass(
                cp, self.optmodel, self.processes, self.pool, self.solpool
            )
        # to device
        if self.solpool.device != cp.device:
            self.solpool = self.solpool.to(cp.device)
        # obj for solpool
        objpool_c = true_cost @ self.solpool.T  # true cost
        objpool_cp = pred_cost @ self.solpool.T  # pred cost
        # cross entropy loss
        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss = -(F.log_softmax(objpool_cp, dim=1) * F.softmax(objpool_c, dim=1).clamp(min=1e-8))
        elif self.optmodel.modelSense == EPO.MAXIMIZE:
            loss = -(
                F.log_softmax(-objpool_cp, dim=1) * F.softmax(-objpool_c, dim=1).clamp(min=1e-8)
            )
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        return self._reduce(loss)


class pairwiseLTR(optModule):
    """
    An autograd module for pairwise learning to rank, where the goal is to learn
    an objective function that ranks a pool of feasible solutions correctly.

    For the pairwise LTR, the cost vector needs to be predicted from the
    contextual data and the loss learns the relative ordering of pairs of items.

    Thus, it allows us to design an algorithm based on stochastic gradient
    descent.

    Reference: <https://proceedings.mlr.press/v162/mandi22a.html>
    """

    def __init__(
        self,
        optmodel: optModel,
        processes: int = 1,
        solve_ratio: float = 1.0,
        reduction: Reduction = "mean",
        dataset: optDataset | None = None,
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            processes: number of processors, 1 for single-core, 0 for all of cores
            solve_ratio: the ratio of new solutions computed during training
            reduction: the reduction to apply to the output
            dataset: the training data
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset, require_solpool=True)
        # function
        self.relu = nn.ReLU()

    def forward(self, pred_cost: torch.Tensor, true_cost: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # convert tensor
        cp = pred_cost.detach()
        # solve and update pool
        if self._branch_rng.uniform() <= self.solve_ratio:
            _, _, self.solpool = _solve_in_pass(
                cp, self.optmodel, self.processes, self.pool, self.solpool
            )
        # to device
        if self.solpool.device != cp.device:
            self.solpool = self.solpool.to(cp.device)
        # obj for solpool
        objpool_c = true_cost @ self.solpool.T  # true cost
        objpool_cp = pred_cost @ self.solpool.T  # pred cost
        # best solutions for each instance
        if self.optmodel.modelSense == EPO.MINIMIZE:
            best_inds = torch.argmin(objpool_c, dim=1)
        elif self.optmodel.modelSense == EPO.MAXIMIZE:
            best_inds = torch.argmax(objpool_c, dim=1)
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        objpool_cp_best = objpool_cp.gather(1, best_inds.unsqueeze(1))
        # best-vs-rest diff
        solpool_size = objpool_cp.shape[1]
        if self.optmodel.modelSense == EPO.MINIMIZE:
            diff = objpool_cp_best - objpool_cp
        else:
            diff = objpool_cp - objpool_cp_best
        # ranking loss; best-vs-best slot contributes 0
        loss = self.relu(diff).sum(dim=1) / max(solpool_size - 1, 1)
        return self._reduce(loss)


class pointwiseLTR(optModule):
    """
    An autograd module for pointwise learning to rank, where the goal is to
    learn an objective function that ranks a pool of feasible solutions
    correctly.

    For the pointwise LTR, the cost vector needs to be predicted from contextual
    data, and calculates the ranking scores of the items.

    Thus, it allows us to design an algorithm based on stochastic gradient
    descent.

    Reference: <https://proceedings.mlr.press/v162/mandi22a.html>
    """

    def __init__(
        self,
        optmodel: optModel,
        processes: int = 1,
        solve_ratio: float = 1.0,
        reduction: Reduction = "mean",
        dataset: optDataset | None = None,
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            processes: number of processors, 1 for single-core, 0 for all of cores
            solve_ratio: the ratio of new solutions computed during training
            reduction: the reduction to apply to the output
            dataset: the training data
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset, require_solpool=True)

    def forward(self, pred_cost: torch.Tensor, true_cost: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # convert tensor
        cp = pred_cost.detach()
        # solve and update pool
        if self._branch_rng.uniform() <= self.solve_ratio:
            _, _, self.solpool = _solve_in_pass(
                cp, self.optmodel, self.processes, self.pool, self.solpool
            )
        # to device
        if self.solpool.device != cp.device:
            self.solpool = self.solpool.to(cp.device)
        # squared loss
        loss = ((true_cost - pred_cost) @ self.solpool.T).square().mean(dim=1)
        return self._reduce(loss)
