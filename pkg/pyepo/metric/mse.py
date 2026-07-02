#!/usr/bin/env python
"""
Mean Squared Error
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pyepo.metric._common import torch_evaluation, validate_prediction_batch

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader


def MSE(predmodel: nn.Module, dataloader: DataLoader) -> float:
    """
    A function to evaluate model performance with MSE

    Args:
        predmodel: a regression neural network for cost prediction
        dataloader: Torch dataloader from optDataSet (fields beyond
            ``(x, c, w, z)`` are ignored)

    Returns:
        float: MSE loss
    """
    loss = 0
    total = 0
    with torch_evaluation(predmodel) as device, torch.no_grad():
        # load data
        for data in dataloader:
            x, c, _, _ = data[:4]
            x, c = x.to(device), c.to(device)
            # predict
            cp = predmodel(x)
            validate_prediction_batch(cp, c)
            loss += ((cp - c) ** 2).mean(dim=1).sum().item()
            total += x.shape[0]
    return loss / total if total else 0.0
