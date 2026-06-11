#!/usr/bin/env python
"""
True regret loss
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import TYPE_CHECKING

import numpy as np
import torch
from pathos.multiprocessing import ProcessingPool

from pyepo import EPO
from pyepo.func.utils import _close_pool, _init_worker_model, _solve_batch
from pyepo.model.mpax import optMpaxModel
from pyepo.utils import _EPS, costToNumpy, getArgs

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader

    from pyepo.model.opt import optModel

logger = logging.getLogger(__name__)


def _regretFromObj(obj, true_obj, modelSense):
    """
    Signed regret gap between a realized objective and the true optimum.

    Operates elementwise on scalars or numpy arrays.

    Args:
        obj: objective value(s) of the evaluated decision under the true cost
        true_obj: true optimal objective value(s)
        modelSense: ``EPO.MINIMIZE`` or ``EPO.MAXIMIZE``

    Returns:
        float or np.ndarray: regret gap(s)
    """
    if modelSense == EPO.MINIMIZE:
        return obj - true_obj
    if modelSense == EPO.MAXIMIZE:
        return true_obj - obj
    raise ValueError("Invalid modelSense.")


def _objOffset(optmodel) -> float:
    """
    Bare objective constant carried by a compiled DSL problem, 0 otherwise.

    Args:
        optmodel: optimization model

    Returns:
        float: objective constant
    """
    problem = getattr(optmodel, "problem", None)
    return float(problem.obj_offset) if problem is not None else 0.0


def regret(
    predmodel: nn.Module,
    optmodel: optModel,
    dataloader: DataLoader,
    processes: int = 1,
    reduction: str = "normalized",
) -> float | np.ndarray:
    """
    True regret (SPO loss) of a trained predictor.

    Solves the optimization problem on the predicted cost vector
    :math:`\\hat{\\mathbf{c}}`, then measures the excess true objective
    incurred by that decision:
    :math:`l_i = \\mathbf{c}_i^\\top \\mathbf{w}^*(\\hat{\\mathbf{c}}_i) -
    z^*(\\mathbf{c}_i)`. With the default ``reduction="normalized"`` the
    result is :math:`\\sum_i l_i / \\sum_i |z^*(\\mathbf{c}_i)|`,
    dimensionless and comparable across problem scales; instances with
    near-zero true optima inflate the ratio. The predictor is evaluated
    under ``eval()``; its original mode is restored afterwards.

    Args:
        predmodel: a regression neural network for cost prediction
        optmodel: a PyEPO optimization model
        dataloader: PyTorch DataLoader over an ``optDataset`` (yielding
            ``(x, c, w, z)`` tuples)
        processes: number of processors, 1 for single-core, 0 for all of
            cores; a fresh worker pool is spawned per call, each worker
            rebuilding the model from its constructor args
        reduction: "normalized" (sum of regrets over sum of absolute true
            optima), "sum", "mean", or "none" (per-instance array)

    Returns:
        float or np.ndarray: aggregated regret, or per-instance regrets
        when ``reduction="none"``
    """
    if reduction not in ("normalized", "sum", "mean", "none"):
        raise ValueError(f"Invalid reduction '{reduction}'.")
    # force processes to 1 for MPAX
    if isinstance(optmodel, optMpaxModel) and processes != 1:
        logger.warning("MPAX does not support multiprocessing. Setting `processes = 1`.")
        processes = 1
    # number of processes
    if processes < 0 or processes > mp.cpu_count():
        raise ValueError(f"Invalid processors number {processes}, only {mp.cpu_count()} cores.")
    processes = mp.cpu_count() if processes == 0 else processes
    losses = []
    optsum = 0.0
    # get device (cpu fallback for parameterless predictors)
    param = next(predmodel.parameters(), None)
    device = param.device if param is not None else torch.device("cpu")
    # multi-core: each worker builds its own optmodel once via the initializer
    pool = None
    if processes > 1:
        pool = ProcessingPool(
            processes,
            initializer=_init_worker_model,
            initargs=(type(optmodel), getArgs(optmodel)),
        )
    # evaluate under eval(); the original mode is restored afterwards
    was_training = predmodel.training
    predmodel.eval()
    try:
        # load data
        for data in dataloader:
            x, c, _, z = data
            x, c, z = x.to(device), c.to(device), z.to(device)
            # predict and batch-solve all instances in one call
            with torch.no_grad():
                cp = predmodel(x)
            # full cost so the MPAX backend can batch-set the objective in one call
            sols, _ = _solve_batch(optmodel._fullCost(cp), optmodel, processes=processes, pool=pool)
            # vectorized regret accumulation (one host sync per batch)
            sols_np = costToNumpy(sols)
            c_np = costToNumpy(optmodel._fullCost(c))
            z_np = costToNumpy(z).reshape(-1)
            # bare objective constants live outside the dot product
            obj = np.einsum("bi,bi->b", sols_np, c_np) + _objOffset(optmodel)
            losses.append(_regretFromObj(obj, z_np, optmodel.modelSense))
            optsum += float(np.abs(z_np).sum())
    finally:
        if pool is not None:
            _close_pool(pool)
        # restore the original mode even if evaluation raises
        predmodel.train(was_training)
    loss = np.concatenate(losses) if losses else np.empty(0)
    # reduce
    if reduction == "normalized":
        return float(loss.sum()) / (optsum + _EPS)
    if reduction == "sum":
        return float(loss.sum())
    if reduction == "mean":
        return float(loss.mean()) if loss.size else 0.0
    return loss


def calRegret(
    optmodel: optModel,
    pred_cost: np.ndarray,
    true_cost: np.ndarray,
    true_obj: float,
) -> float:
    """
    True regret of a single instance.

    Args:
        optmodel: optimization model
        pred_cost: predicted cost vector
        true_cost: true cost vector
        true_obj: true optimal objective value

    Returns:
        float: true regret
    """
    # opt sol for pred cost
    optmodel.setObj(optmodel._fullCost(pred_cost))
    sol, _ = optmodel.solve()
    # MPAX backend returns a torch tensor; convert without coercing dtype
    if isinstance(sol, torch.Tensor):
        sol = sol.detach().cpu().numpy()
    # full objective of the predicted decision at the true cost
    obj = np.dot(sol, optmodel._fullCost(np.asarray(true_cost, dtype=float))) + _objOffset(optmodel)
    return float(_regretFromObj(obj, true_obj, optmodel.modelSense))
