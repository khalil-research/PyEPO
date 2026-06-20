#!/usr/bin/env python
"""
Learning to rank Losses
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pyepo import EPO
from pyepo.func.jax.abcmodule import optModule
from pyepo.func.jax.utils import _full_cost


class listwiseLearningToRank(optModule):
    """
    Listwise Learning-to-Rank loss over a cached solution pool.

    Models the ranking distribution over the pool as a SoftMax of
    predicted-cost scores and minimizes its cross-entropy against the true
    ranking distribution.

    Reference: Mandi et al. (2022)
    `<https://proceedings.mlr.press/v162/mandi22a.html>`_
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1.0, reduction="mean", dataset=None):
        """
        Args:
            optmodel: a PyEPO optimization model
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of instances solved exactly each step
            reduction: reduction applied to the batch loss ("mean", "sum", "none")
            dataset: training dataset used to seed the solution pool
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset, require_solpool=True)

    def forward(self, pred_cost, true_cost):
        """
        Forward pass
        """
        # lift to the full objective space
        pred_cost = _full_cost(pred_cost, self.optmodel)
        true_cost = _full_cost(true_cost, self.optmodel)
        # solve and update pool
        solpool = self._refresh_solution_pool(pred_cost)
        # obj for solpool
        objpool_c = true_cost @ solpool.T
        objpool_cp = pred_cost @ solpool.T
        # cross entropy loss, summed over the pool per instance
        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss = -(
                jax.nn.log_softmax(-objpool_cp, axis=1)
                * jnp.clip(jax.nn.softmax(-objpool_c, axis=1), 1e-8, None)
            ).sum(axis=1)
        else:
            loss = -(
                jax.nn.log_softmax(objpool_cp, axis=1)
                * jnp.clip(jax.nn.softmax(objpool_c, axis=1), 1e-8, None)
            ).sum(axis=1)
        return self._reduce(loss)


class pairwiseLearningToRank(optModule):
    """
    Pairwise Learning-to-Rank loss over a cached solution pool.

    Enforces a margin between the true optimum (best pool member) and each
    suboptimal solution via a ReLU hinge on the predicted-cost difference.

    Reference: Mandi et al. (2022)
    `<https://proceedings.mlr.press/v162/mandi22a.html>`_
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1.0, reduction="mean", dataset=None):
        """
        Args:
            optmodel: a PyEPO optimization model
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of instances solved exactly each step
            reduction: reduction applied to the batch loss ("mean", "sum", "none")
            dataset: training dataset used to seed the solution pool
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset, require_solpool=True)

    def forward(self, pred_cost, true_cost):
        """
        Forward pass
        """
        # lift to the full objective space
        pred_cost = _full_cost(pred_cost, self.optmodel)
        true_cost = _full_cost(true_cost, self.optmodel)
        # solve and update pool
        solpool = self._refresh_solution_pool(pred_cost)
        # obj for solpool
        objpool_c = true_cost @ solpool.T
        objpool_cp = pred_cost @ solpool.T
        # best solution per instance
        if self.optmodel.modelSense == EPO.MINIMIZE:
            best_inds = jnp.argmin(objpool_c, axis=1)
        else:
            best_inds = jnp.argmax(objpool_c, axis=1)
        objpool_cp_best = jnp.take_along_axis(objpool_cp, best_inds[:, None], axis=1)
        # best-vs-rest diff
        if self.optmodel.modelSense == EPO.MINIMIZE:
            diff = objpool_cp_best - objpool_cp
        else:
            diff = objpool_cp - objpool_cp_best
        # ranking loss; best-vs-best slot contributes 0
        loss = jax.nn.relu(diff).sum(axis=1) / max(solpool.shape[0] - 1, 1)
        return self._reduce(loss)


class pointwiseLearningToRank(optModule):
    """
    Pointwise Learning-to-Rank loss over a cached solution pool.

    Fits the predicted score of each pool member toward its true score by
    squared error, averaged over the pool.

    Reference: Mandi et al. (2022)
    `<https://proceedings.mlr.press/v162/mandi22a.html>`_
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1.0, reduction="mean", dataset=None):
        """
        Args:
            optmodel: a PyEPO optimization model
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of instances solved exactly each step
            reduction: reduction applied to the batch loss ("mean", "sum", "none")
            dataset: training dataset used to seed the solution pool
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset, require_solpool=True)

    def forward(self, pred_cost, true_cost):
        """
        Forward pass
        """
        # lift to the full objective space
        pred_cost = _full_cost(pred_cost, self.optmodel)
        true_cost = _full_cost(true_cost, self.optmodel)
        # solve and update pool
        solpool = self._refresh_solution_pool(pred_cost)
        # squared loss over the pool
        loss = (((true_cost - pred_cost) @ solpool.T) ** 2).mean(axis=1)
        return self._reduce(loss)


# acronym aliases
lsLTR = listwiseLearningToRank
prLTR = pairwiseLearningToRank
ptLTR = pointwiseLearningToRank
