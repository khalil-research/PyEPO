#!/usr/bin/env python
"""
True regret loss
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from pyepo import EPO
from pyepo.func.utils import _solve_batch
from pyepo.utils import _EPS, costToNumpy

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader

    from pyepo.model.opt import optModel


def regret(
    predmodel: nn.Module,
    optmodel: optModel,
    dataloader: DataLoader,
) -> float:
    """
    Normalized true regret (SPO loss) of a trained predictor.

    Solves the optimization problem on the predicted cost vector
    :math:`\\hat{\\mathbf{c}}`, then measures the excess true objective
    incurred by that decision:
    :math:`\\sum_i (\\mathbf{c}_i^\\top \\mathbf{w}^*(\\hat{\\mathbf{c}}_i) -
    z^*(\\mathbf{c}_i)) / \\sum_i |z^*(\\mathbf{c}_i)|`. The result is
    dimensionless and comparable across problem scales. The predictor is
    automatically put into ``eval()`` for the call and restored to
    ``train()`` afterwards.

    Args:
        predmodel: a regression neural network for cost prediction
        optmodel: a PyEPO optimization model
        dataloader: PyTorch DataLoader over an ``optDataset`` (yielding
            ``(x, c, w, z)`` tuples)

    Returns:
        float: normalized true regret
    """
    # evaluate
    predmodel.eval()
    loss = 0.0
    optsum = 0.0
    # get device
    device = next(predmodel.parameters()).device
    try:
        # load data
        for data in dataloader:
            x, c, _, z = data
            x, c, z = x.to(device), c.to(device), z.to(device)
            # predict and batch-solve all instances in one call
            with torch.no_grad():
                cp = predmodel(x)
            # full cost so the MPAX backend can batch-set the objective in one call
            sols, _ = _solve_batch(optmodel._fullCost(cp), optmodel, processes=1, pool=None)
            # vectorized regret accumulation (one host sync per batch)
            sols_np = costToNumpy(sols)
            c_np = costToNumpy(optmodel._fullCost(c))
            z_np = costToNumpy(z).reshape(-1)
            obj = np.einsum("bi,bi->b", sols_np, c_np)
            if optmodel.modelSense == EPO.MINIMIZE:
                batch_loss = obj - z_np
            elif optmodel.modelSense == EPO.MAXIMIZE:
                batch_loss = z_np - obj
            else:
                raise ValueError("Invalid modelSense.")
            loss += float(batch_loss.sum())
            optsum += float(np.abs(z_np).sum())
    finally:
        # restore training mode even if evaluation raises
        predmodel.train()
    # normalized
    return loss / (optsum + _EPS)


def calRegret(
    optmodel: optModel,
    pred_cost: np.ndarray,
    true_cost: np.ndarray,
    true_obj: float,
) -> float:
    """
    A function to calculate normalized true regret for a batch

    Args:
        optmodel: optimization model
        pred_cost: predicted costs
        true_cost: true costs
        true_obj: true optimal objective value

    Returns:
        float: true regret loss
    """
    # opt sol for pred cost
    optmodel.setObj(pred_cost)
    sol, _ = optmodel.solve()
    # MPAX backend returns a torch tensor; convert without coercing dtype
    if isinstance(sol, torch.Tensor):
        sol = sol.detach().cpu().numpy()
    # full objective of the predicted decision at the true cost
    obj = np.dot(sol, optmodel._fullCost(np.asarray(true_cost, dtype=float)))
    # loss
    if optmodel.modelSense == EPO.MINIMIZE:
        loss = obj - true_obj
    elif optmodel.modelSense == EPO.MAXIMIZE:
        loss = true_obj - obj
    else:
        raise ValueError("Invalid modelSense.")
    return loss
