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
from pyepo.metric._common import (
    normalize_regret,
    torch_evaluation,
    validate_cost_vectors,
    validate_prediction_batch,
    validate_retry_count,
    validate_tolerance,
)
from pyepo.metric.regret import _checkLinearObj, _objOffset, _regretFromObj
from pyepo.utils import costToNumpy

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
    max_iter: int = 10,
) -> float:
    """
    Normalized unambiguous regret (worst-case-tie SPO loss).

    When the predicted cost vector :math:`\\hat{\\mathbf{c}}` yields multiple
    optimal solutions, ``regret`` reports the realized one while
    ``unambRegret`` reports the **worst case over all optima**:
    :math:`l_{\\mathrm{URegret}}(\\hat{\\mathbf{c}}, \\mathbf{c}) =
    \\max_{\\mathbf{w} \\in W^*(\\hat{\\mathbf{c}})} \\mathbf{c}^\\top
    \\mathbf{w} - z^*(\\mathbf{c})`. More theoretically rigorous than
    ``regret``, but in practice the two are nearly identical and
    ``unambRegret`` is rarely required.
    The result is normalized by :math:`\\sum_i |z^*(\\mathbf{c}_i)|`; instances with near-zero true optima inflate the ratio.

    Args:
        predmodel: a regression neural network for cost prediction
        optmodel: a PyEPO optimization model
        dataloader: PyTorch DataLoader over an ``optDataset``
        tolerance: precision used when rounding predicted costs to find ties
        max_iter: maximum number of solve retries with relaxed tolerance

    Returns:
        float: normalized unambiguous regret
    """
    validate_tolerance(tolerance)
    validate_retry_count(max_iter)
    _checkLinearObj(optmodel)
    loss = 0
    optsum = 0
    with torch_evaluation(predmodel) as device:
        # load data
        for data in dataloader:
            x, c, w, z = data
            x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
            # pred cost
            with torch.no_grad():
                cp = costToNumpy(predmodel(x))
            c_np = costToNumpy(c)
            validate_prediction_batch(cp, c_np, optmodel.num_cost)
            # solve
            for j in range(cp.shape[0]):
                # accumulate loss
                loss += calUnambRegret(
                    optmodel, cp[j], c_np[j], z[j].item(), tolerance, max_iter=max_iter
                )
            optsum += abs(z).sum().item()
    # normalized
    return normalize_regret(loss, optsum)


def calUnambRegret(
    optmodel: optModel,
    pred_cost: np.ndarray,
    true_cost: np.ndarray,
    true_obj: float,
    tolerance: float = 1e-5,
    max_iter: int = 10,
) -> float:
    """
    Unambiguous (worst-case-tie) regret of a single instance.

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
    validate_tolerance(tolerance)
    validate_retry_count(max_iter)
    pred_cost, true_cost, true_obj = validate_cost_vectors(
        pred_cost, true_cost, true_obj, optmodel.num_cost
    )
    _checkLinearObj(optmodel)
    # lift to the full objective space, then change precision
    cp = np.around(optmodel._fullCost(np.asarray(pred_cost, dtype=float)) / tolerance)
    # opt sol for pred cost
    optmodel._setFullObj(cp)
    sol, _ = optmodel.solve()
    # MPAX backend may return a torch tensor; convert without dtype coercion
    if isinstance(sol, torch.Tensor):
        sol = sol.detach().cpu().numpy()
    # loose-side rounding keeps the optimum in the tie set
    if optmodel.modelSense == EPO.MINIMIZE:
        objp = np.ceil(np.dot(cp, np.asarray(sol).T))
        wst_optmodel = optmodel.addConstr(cp, objp + 1e-2)
    elif optmodel.modelSense == EPO.MAXIMIZE:
        objp = np.floor(np.dot(cp, np.asarray(sol).T))
        wst_optmodel = optmodel.addConstr(-cp, -objp + 1e-2)
    else:
        raise ValueError("Invalid modelSense.")
    # opt model to find worst case
    c_full = optmodel._fullCost(np.asarray(true_cost, dtype=float))
    try:
        wst_optmodel._setFullObj(-c_full)
        wst_sol, _ = wst_optmodel.solve()
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
    # MPAX backend may return a torch tensor; convert without dtype coercion
    if isinstance(wst_sol, torch.Tensor):
        wst_sol = wst_sol.detach().cpu().numpy()
    # worst-case full objective of the tied decisions at the true cost
    obj = np.dot(np.asarray(wst_sol), c_full) + _objOffset(optmodel)
    return float(_regretFromObj(obj, true_obj, optmodel.modelSense))
