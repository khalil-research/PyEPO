#!/usr/bin/env python
# coding: utf-8
"""Shared pytest fixtures and helpers for PyEPO tests."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader


try:
    from pyepo.model.grb.shortestpath import shortestPathModel  # noqa: F401
    _HAS_GUROBI = True
except (ImportError, NameError):
    _HAS_GUROBI = False

_HAS_CUDA = torch.cuda.is_available()


NUM_DATA = 32
NUM_FEAT = 3
GRID = (3, 3)
BATCH = 16


class LinearPred(nn.Module):
    """Minimal linear pred model: features -> per-edge cost."""

    def __init__(self, num_feat, num_cost):
        super().__init__()
        self.linear = nn.Linear(num_feat, num_cost)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture(scope="module")
def sp_data():
    """Shortest-path dataset + loader (CPU). Skipped if Gurobi missing."""
    if not _HAS_GUROBI:
        pytest.skip("Gurobi not installed")
    import pyepo
    from pyepo.data.dataset import optDataset
    from pyepo.model.grb.shortestpath import shortestPathModel

    x, c = pyepo.data.shortestpath.genData(NUM_DATA, NUM_FEAT, GRID, seed=42)
    optmodel = shortestPathModel(grid=GRID)
    dataset = optDataset(optmodel, x, c)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)
    return optmodel, dataset, loader


@pytest.fixture(scope="module")
def ks_data():
    """Knapsack dataset + loader (CPU). Skipped if Gurobi missing."""
    if not _HAS_GUROBI:
        pytest.skip("Gurobi not installed")
    import pyepo
    from pyepo.data.dataset import optDataset
    from pyepo.model.grb.knapsack import knapsackModel

    weights, x, c = pyepo.data.knapsack.genData(
        NUM_DATA, NUM_FEAT, 4, dim=1, deg=1, seed=42)
    optmodel = knapsackModel(weights=weights, capacity=[10.0])
    dataset = optDataset(optmodel, x, c)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)
    return optmodel, dataset, loader
