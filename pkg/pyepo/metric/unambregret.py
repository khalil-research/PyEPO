#!/usr/bin/env python
"""
Unambiguous regret loss
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from pyepo import EPO
from pyepo.utils import _EPS, costToNumpy

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader

    from pyepo.model.opt import optModel

logger = logging.getLogger(__name__)


def unambRegret(
    predmodel: nn.Module,
    optmodel: optModel,
    dataloader: DataLoader,
    tolerance: float = 1e-5,
) -> float:
    """
    A function to evaluate model performance with normalized unambiguous regret

    Args:
        predmodel: a regression neural network for cost prediction
        optmodel: a PyEPO optimization model
        dataloader: Torch dataloader from optDataSet
        tolerance: tolerance for optimization

    Returns:
        float: unambiguous regret loss
    """
    # evaluate
    predmodel.eval()
    loss = 0
    optsum = 0
    # get device
    device = next(predmodel.parameters()).device
    try:
        # load data
        for data in dataloader:
            x, c, w, z = data
            x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
            # pred cost
            with torch.no_grad():
                cp = costToNumpy(predmodel(x))
            c_np = costToNumpy(c)
            # solve
            for j in range(cp.shape[0]):
                # accumulate loss
                loss += calUnambRegret(
                    optmodel, cp[j], c_np[j], z[j].item(), tolerance, max_iter=10
                )
            optsum += abs(z).sum().item()
    finally:
        # restore training mode even if evaluation raises
        predmodel.train()
    # normalized
    return loss / (optsum + _EPS)


def calUnambRegret(
    optmodel: optModel,
    pred_cost: np.ndarray,
    true_cost: np.ndarray,
    true_obj: float,
    tolerance: float = 1e-5,
    max_iter: int = 10,
) -> float:
    """
    A function to calculate normalized unambiguous regret for a batch

    Args:
        optmodel: optimization model
        pred_cost: predicted costs
        true_cost: true costs
        true_obj: true optimal objective value
        tolerance: tolerance for precision
        max_iter: maximum number of recursive retries

    Returns:
        float: unambiguous regret loss
    """
    if max_iter <= 0:
        raise RuntimeError("Max iterations reached in calUnambRegret.")
    # change precision
    cp = np.around(pred_cost / tolerance).astype(int)
    # opt sol for pred cost
    optmodel.setObj(cp)
    sol, objp = optmodel.solve()
    # MPAX backend may return a torch tensor; convert without dtype coercion
    if isinstance(sol, torch.Tensor):
        sol = sol.detach().cpu().numpy()
    objp = np.ceil(np.dot(cp, np.asarray(sol).T))
    # opt for pred cost
    if optmodel.modelSense == EPO.MINIMIZE:
        wst_optmodel = optmodel.addConstr(cp, objp + 1e-2)
    elif optmodel.modelSense == EPO.MAXIMIZE:
        wst_optmodel = optmodel.addConstr(-cp, -objp + 1e-2)
    else:
        raise ValueError("Invalid modelSense.")
    # opt model to find worst case
    try:
        wst_optmodel.setObj(-true_cost)
        _, obj = wst_optmodel.solve()
    except Exception as e:  # noqa: BLE001  any solver failure triggers retry
        new_tolerance = tolerance * 10
        logger.warning(
            "calUnambRegret: solve failed (%s: %s); retrying with tolerance=%g (%d retries left)",
            type(e).__name__,
            e,
            new_tolerance,
            max_iter - 1,
        )
        return calUnambRegret(
            optmodel, pred_cost, true_cost, true_obj, tolerance=new_tolerance, max_iter=max_iter - 1
        )
    obj = -obj
    # loss
    if optmodel.modelSense == EPO.MINIMIZE:
        loss = obj - true_obj
    elif optmodel.modelSense == EPO.MAXIMIZE:
        loss = true_obj - obj
    else:
        raise ValueError("Invalid modelSense.")
    return loss
