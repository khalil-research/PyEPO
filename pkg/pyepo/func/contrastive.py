#!/usr/bin/env python
"""
Noise contrastive estimation loss function
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pyepo import EPO
from pyepo.func.abcmodule import optModule

if TYPE_CHECKING:
    from pyepo.data.dataset import optDataset
    from pyepo.func.abcmodule import Reduction
    from pyepo.model.opt import optModel


class noiseContrastiveEstimation(optModule):
    """
    Noise Contrastive Estimation (NCE) -- contrastive loss against a cached solution pool.

    Averages the predicted-cost margin between the true optimum and every
    member of the cached pool :math:`\\Gamma`:
    :math:`\\mathcal{L} = \\tfrac{1}{|\\Gamma|}\\sum_{\\mathbf{w} \\in \\Gamma}
    (\\hat{\\mathbf{c}}^\\top \\mathbf{w}^*(\\mathbf{c}) -
    \\hat{\\mathbf{c}}^\\top \\mathbf{w})`. The gradient has a closed form
    (no solver call in the backward pass), so per-step cost is dominated by
    occasional pool refreshes rather than by solver work. Pass
    ``solve_ratio < 1`` to control refresh frequency; the pool is seeded
    from ``dataset`` at construction.

    Reference: Mulamba et al. (2021)
    `<https://www.ijcai.org/proceedings/2021/390>`_
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
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: probability of refreshing the pool each batch (1.0 = always solve)
            reduction: reduction applied to the batch loss (``"mean"``, ``"sum"``, ``"none"``)
            dataset: training dataset used to seed the solution pool (required)
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset, require_solpool=True)

    def forward(self, pred_cost: torch.Tensor, true_sol: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # lift cost to the full objective space (no-op without partial prediction)
        pred_cost = self.optmodel._fullCost(pred_cost)
        # convert tensor
        cp = pred_cost.detach()
        solpool = self._refresh_solution_pool(cp)
        # get current obj
        obj_cp = torch.einsum("bd,bd->b", pred_cost, true_sol).unsqueeze(1)
        # get obj for solpool
        objpool_cp = torch.einsum("bd,nd->bn", pred_cost, solpool)
        # get loss
        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss = (obj_cp - objpool_cp).mean(dim=1)
        else:
            loss = (objpool_cp - obj_cp).mean(dim=1)
        return self._reduce(loss)


class contrastiveMAP(optModule):
    """
    Contrastive Maximum-a-Posteriori (CMAP) -- max-margin special case of NCE.

    Keeps only the most-violating member of the cached pool :math:`\\Gamma`
    (the one with the smallest predicted-cost objective) as the negative:
    :math:`\\mathcal{L} = \\hat{\\mathbf{c}}^\\top \\mathbf{w}^*(\\mathbf{c}) -
    \\min_{\\mathbf{w} \\in \\Gamma} \\hat{\\mathbf{c}}^\\top \\mathbf{w}`.
    Simpler than NCE and often equally effective. Pool semantics
    (``solve_ratio``, ``dataset``) are identical to NCE.

    Reference: Mulamba et al. (2021)
    `<https://www.ijcai.org/proceedings/2021/390>`_
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
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: probability of refreshing the pool each batch (1.0 = always solve)
            reduction: reduction applied to the batch loss (``"mean"``, ``"sum"``, ``"none"``)
            dataset: training dataset used to seed the solution pool (required)
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset, require_solpool=True)

    def forward(self, pred_cost: torch.Tensor, true_sol: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # lift cost to the full objective space (no-op without partial prediction)
        pred_cost = self.optmodel._fullCost(pred_cost)
        # convert tensor
        cp = pred_cost.detach()
        solpool = self._refresh_solution_pool(cp)
        # get current obj
        obj_cp = torch.einsum("bd,bd->b", pred_cost, true_sol).unsqueeze(1)
        # get obj for solpool
        objpool_cp = torch.einsum("bd,nd->bn", pred_cost, solpool)
        # get loss
        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss, _ = (obj_cp - objpool_cp).max(dim=1)
        else:
            loss, _ = (objpool_cp - obj_cp).max(dim=1)
        return self._reduce(loss)


# acronym aliases
NCE = noiseContrastiveEstimation
CMAP = contrastiveMAP
