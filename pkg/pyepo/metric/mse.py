#!/usr/bin/env python
# coding: utf-8
"""
Mean Squared Error
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader


def MSE(predmodel: nn.Module, dataloader: DataLoader) -> float:
    """
    A function to evaluate model performance with MSE

    Args:
        predmodel (nn): a regression neural network for cost prediction
        dataloader (DataLoader): Torch dataloader from optDataSet

    Returns:
        float: MSE loss
    """
    # evaluate
    predmodel.eval()
    loss = 0
    # get device
    device = next(predmodel.parameters()).device
    try:
        # load data
        with torch.no_grad():
            for data in dataloader:
                x, c, w, z = data
                x, c = x.to(device), c.to(device)
                # predict
                cp = predmodel(x)
                loss += ((cp - c) ** 2).mean(dim=1).sum().item()
    finally:
        # restore training mode even if evaluation raises
        predmodel.train()
    return loss / len(dataloader.dataset)
