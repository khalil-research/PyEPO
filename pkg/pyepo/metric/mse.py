#!/usr/bin/env python
"""
Mean Squared Error
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

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
    # evaluate under eval(); the original mode is restored afterwards
    was_training = predmodel.training
    predmodel.eval()
    loss = 0
    # get device (cpu fallback for parameterless predictors)
    param = next(predmodel.parameters(), None)
    device = param.device if param is not None else torch.device("cpu")
    try:
        # load data
        with torch.no_grad():
            for data in dataloader:
                x, c, _, _ = data
                x, c = x.to(device), c.to(device)
                # predict
                cp = predmodel(x)
                loss += ((cp - c) ** 2).mean(dim=1).sum().item()
    finally:
        # restore the original mode even if evaluation raises
        predmodel.train(was_training)
    return loss / len(cast("Sized", dataloader.dataset))
