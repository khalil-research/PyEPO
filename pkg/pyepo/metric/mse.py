#!/usr/bin/env python
"""
Mean Squared Error
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from pyepo.metric._common import torch_evaluation

if TYPE_CHECKING:
    from collections.abc import Sized

    from torch import nn
    from torch.utils.data import DataLoader


def MSE(predmodel: nn.Module, dataloader: DataLoader) -> float:
    """
    A function to evaluate model performance with MSE

    Args:
        predmodel: a regression neural network for cost prediction
        dataloader: Torch dataloader from optDataSet

    Returns:
        float: MSE loss
    """
    loss = 0
    with torch_evaluation(predmodel) as device, torch.no_grad():
        # load data
        for data in dataloader:
            x, c, _, _ = data
            x, c = x.to(device), c.to(device)
            # predict
            cp = predmodel(x)
            loss += ((cp - c) ** 2).mean(dim=1).sum().item()
    return loss / len(cast("Sized", dataloader.dataset))
