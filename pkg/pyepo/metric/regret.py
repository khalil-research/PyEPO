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
    A function to evaluate model performance with normalized true regret

    Args:
        predmodel: a regression neural network for cost prediction
        optmodel: a PyEPO optimization model
        dataloader: Torch dataloader from optDataSet

    Returns:
        float: true regret loss
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
            sols, _ = _solve_batch(cp, optmodel, processes=1, pool=None)
            # vectorized regret accumulation (one host sync per batch)
            sols_np = costToNumpy(sols)
            c_np = costToNumpy(c)
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
    # solver may return a torch tensor (MPAX backend) — convert without
    # auto-casting Python lists to float32 (would silently downcast regret)
    if isinstance(sol, torch.Tensor):
        sol = sol.detach().cpu().numpy()
    # obj with true cost
    obj = np.dot(sol, true_cost)
    # loss
    if optmodel.modelSense == EPO.MINIMIZE:
        loss = obj - true_obj
    elif optmodel.modelSense == EPO.MAXIMIZE:
        loss = true_obj - obj
    else:
        raise ValueError("Invalid modelSense.")
    return loss
