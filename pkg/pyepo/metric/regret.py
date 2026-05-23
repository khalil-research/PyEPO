#!/usr/bin/env python
"""
True regret loss
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from pyepo import EPO
from pyepo.model.opt import costToNumpy

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
        predmodel (nn): a regression neural network for cost prediction
        optmodel (optModel): a PyEPO optimization model
        dataloader (DataLoader): Torch dataloader from optDataSet

    Returns:
        float: true regret loss
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
            # predict
            with torch.no_grad():
                cp = costToNumpy(predmodel(x))
            c_np = costToNumpy(c)
            # solve
            for j in range(cp.shape[0]):
                # accumulate loss
                loss += calRegret(optmodel, cp[j], c_np[j], z[j].item())
            optsum += abs(z).sum().item()
    finally:
        # restore training mode even if evaluation raises
        predmodel.train()
    # normalized
    return loss / (optsum + 1e-7)


def calRegret(
    optmodel: optModel,
    pred_cost: np.ndarray,
    true_cost: np.ndarray,
    true_obj: float,
) -> float:
    """
    A function to calculate normalized true regret for a batch

    Args:
        optmodel (optModel): optimization model
        pred_cost (np.ndarray): predicted costs
        true_cost (np.ndarray): true costs
        true_obj (float): true optimal objective value

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
