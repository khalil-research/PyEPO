#!/usr/bin/env python
"""
Noise contrastive estimation loss function
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from pyepo.func._common import is_minimize
from pyepo.func.jax.abcmodule import optModule
from pyepo.func.jax.utils import _full_cost

if TYPE_CHECKING:
    from pyepo.func.runtime import Reduction


class noiseContrastiveEstimation(optModule):
    """
    Noise Contrastive Estimation (NCE) -- contrastive loss against a cached pool.

    Averages the predicted-cost margin between the true optimum and every pool
    member.

    Reference: Mulamba et al. (2021) `<https://www.ijcai.org/proceedings/2021/390>`_
    """

    def __init__(
        self, optmodel, processes=1, solve_ratio=1.0, reduction: Reduction = "mean", dataset=None
    ):
        """
        Args:
            optmodel: a PyEPO optimization model
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of instances solved exactly each step
            reduction: reduction applied to the batch loss ("mean", "sum", "none")
            dataset: training dataset used to seed the solution pool
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset, require_solpool=True)

    def forward(self, pred_cost, true_sol):
        """
        Forward pass
        """
        # lift to the full objective space
        pred_cost = _full_cost(pred_cost, self.optmodel)
        # solve and update pool
        solpool = self._refresh_solution_pool(pred_cost)
        # current obj and pool obj
        obj_cp = jnp.einsum("bd,bd->b", pred_cost, true_sol)[:, None]
        objpool_cp = jnp.einsum("bd,nd->bn", pred_cost, solpool)
        # margin loss
        if is_minimize(self.optmodel.modelSense):
            loss = (obj_cp - objpool_cp).mean(axis=1)
        else:
            loss = (objpool_cp - obj_cp).mean(axis=1)
        return self._reduce(loss)


class contrastiveMAP(optModule):
    """
    Contrastive Maximum-a-Posteriori (CMAP) -- max-margin special case of NCE.

    Keeps only the most-violating pool member (smallest predicted-cost
    objective) as the negative.

    Reference: Mulamba et al. (2021) `<https://www.ijcai.org/proceedings/2021/390>`_
    """

    def __init__(
        self, optmodel, processes=1, solve_ratio=1.0, reduction: Reduction = "mean", dataset=None
    ):
        """
        Args:
            optmodel: a PyEPO optimization model
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of instances solved exactly each step
            reduction: reduction applied to the batch loss ("mean", "sum", "none")
            dataset: training dataset used to seed the solution pool
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset, require_solpool=True)

    def forward(self, pred_cost, true_sol):
        """
        Forward pass
        """
        # lift to the full objective space
        pred_cost = _full_cost(pred_cost, self.optmodel)
        # solve and update pool
        solpool = self._refresh_solution_pool(pred_cost)
        # current obj and pool obj
        obj_cp = jnp.einsum("bd,bd->b", pred_cost, true_sol)[:, None]
        objpool_cp = jnp.einsum("bd,nd->bn", pred_cost, solpool)
        # max-margin loss
        if is_minimize(self.optmodel.modelSense):
            loss = (obj_cp - objpool_cp).max(axis=1)
        else:
            loss = (objpool_cp - obj_cp).max(axis=1)
        return self._reduce(loss)


# acronym aliases
NCE = noiseContrastiveEstimation
CMAP = contrastiveMAP
