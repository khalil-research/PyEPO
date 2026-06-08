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
    Listwise Learning-to-Rank loss over a cached solution pool.

    Models the ranking distribution over the cached pool :math:`\\Gamma` as
    SoftMax of predicted-cost scores and minimizes its cross-entropy against
    the true ranking distribution. The full-list formulation captures
    interactions between every pair of solutions in :math:`\\Gamma`.
    Pool semantics (``solve_ratio``, ``dataset``) are shared with the other
    LTR variants and with the contrastive methods.

    Reference: Mandi et al. (2022)
    `<https://proceedings.mlr.press/v162/mandi22a.html>`_
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
        # lift costs to the full objective space (no-op without partial prediction)
        pred_cost = self.optmodel._fullCost(pred_cost)
        true_cost = self.optmodel._fullCost(true_cost)
        # convert tensor
        cp = pred_cost.detach()
        # solve and update pool
        if self._branch_rng.uniform() <= self.solve_ratio:
            _, _, self.solpool = _solve_in_pass(
                cp, self.optmodel, self.processes, self.pool, self.solpool
            )
        # require_solpool=True ensures the pool was populated in __init__
        assert self.solpool is not None
        # to device
        if self.solpool.device != cp.device:
            self.solpool = self.solpool.to(cp.device)
        # obj for solpool
        objpool_c = true_cost @ self.solpool.T  # true cost
        objpool_cp = pred_cost @ self.solpool.T  # pred cost
        # cross entropy loss
        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss = -(F.log_softmax(objpool_cp, dim=1) * F.softmax(objpool_c, dim=1).clamp(min=1e-8))
        else:
            loss = -(
                F.log_softmax(-objpool_cp, dim=1) * F.softmax(-objpool_c, dim=1).clamp(min=1e-8)
            )
        return self._reduce(loss)


class pairwiseLTR(optModule):
    """
    Pairwise Learning-to-Rank loss over a cached solution pool.

    Enforces a margin between the true optimum (the best member of
    :math:`\\Gamma`) and each suboptimal solution via a ReLU hinge on the
    predicted-cost difference. Lighter than the listwise variant
    (no SoftMax over the full pool) and often a good first choice when the
    pool is large. Pool semantics (``solve_ratio``, ``dataset``) are shared
    with the other LTR variants and with the contrastive methods.

    Reference: Mandi et al. (2022)
    `<https://proceedings.mlr.press/v162/mandi22a.html>`_
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
        # lift costs to the full objective space (no-op without partial prediction)
        pred_cost = self.optmodel._fullCost(pred_cost)
        true_cost = self.optmodel._fullCost(true_cost)
        # convert tensor
        cp = pred_cost.detach()
        # solve and update pool
        if self._branch_rng.uniform() <= self.solve_ratio:
            _, _, self.solpool = _solve_in_pass(
                cp, self.optmodel, self.processes, self.pool, self.solpool
            )
        # require_solpool=True ensures the pool was populated in __init__
        assert self.solpool is not None
        # to device
        if self.solpool.device != cp.device:
            self.solpool = self.solpool.to(cp.device)
        # obj for solpool
        objpool_c = true_cost @ self.solpool.T  # true cost
        objpool_cp = pred_cost @ self.solpool.T  # pred cost
        # best solutions for each instance
        if self.optmodel.modelSense == EPO.MINIMIZE:
            best_inds = torch.argmin(objpool_c, dim=1)
        else:
            best_inds = torch.argmax(objpool_c, dim=1)
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
    Pointwise Learning-to-Rank loss over a cached solution pool.

    Treats each cached solution :math:`\\mathbf{w} \\in \\Gamma` as an
    independent regression target: the predicted score
    :math:`\\hat{\\mathbf{c}}^\\top \\mathbf{w}` is fit toward the true
    score :math:`\\mathbf{c}^\\top \\mathbf{w}` via squared error, averaged
    over the pool. Cheapest of the three LTR variants -- no cross-pool
    interactions. Pool semantics (``solve_ratio``, ``dataset``) are shared
    with the other LTR variants and with the contrastive methods.

    Reference: Mandi et al. (2022)
    `<https://proceedings.mlr.press/v162/mandi22a.html>`_
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
        # lift costs to the full objective space (no-op without partial prediction)
        pred_cost = self.optmodel._fullCost(pred_cost)
        true_cost = self.optmodel._fullCost(true_cost)
        # convert tensor
        cp = pred_cost.detach()
        # solve and update pool
        if self._branch_rng.uniform() <= self.solve_ratio:
            _, _, self.solpool = _solve_in_pass(
                cp, self.optmodel, self.processes, self.pool, self.solpool
            )
        # require_solpool=True ensures the pool was populated in __init__
        assert self.solpool is not None
        # to device
        if self.solpool.device != cp.device:
            self.solpool = self.solpool.to(cp.device)
        # squared loss
        loss = ((true_cost - pred_cost) @ self.solpool.T).square().mean(dim=1)
        return self._reduce(loss)
