#!/usr/bin/env python
# coding: utf-8
"""
Tests for pyepo.metric: MSE, regret, unambiguous regret, SPOError

Uses small models / mock to keep tests fast (no training).
"""

import pytest
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pyepo import EPO
from pyepo.metric.mse import MSE
from pyepo.metric.regret import calRegret, regret
from pyepo.metric.unambregret import calUnambRegret
from pyepo.metric.metrics import SPOError, testMSE as _testMSE

try:
    from pyepo.model.grb.knapsack import knapsackModel
    from pyepo.model.grb.shortestpath import shortestPathModel
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

requires_gurobi = pytest.mark.skipif(not _HAS_GUROBI, reason="Gurobi not installed")


# ============================================================
# Helpers
# ============================================================

class _IdentityModel(nn.Module):
    """A 'prediction model' that returns its input unchanged."""
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x


class _ConstantModel(nn.Module):
    """A prediction model that always returns a constant vector."""
    def __init__(self, value):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0))  # so parameters() is non-empty
        self.value = value

    def forward(self, x):
        return self.value.expand(x.shape[0], -1)


def _make_dataloader(n=16, d=4, batch_size=8):
    """Create a small DataLoader with (feats, costs, sols, objs)."""
    feats = torch.randn(n, d)
    costs = torch.randn(n, d)
    sols = torch.randn(n, d)
    objs = torch.randn(n, 1)
    ds = TensorDataset(feats, costs, sols, objs)
    return DataLoader(ds, batch_size=batch_size)


# ============================================================
# MSE tests
# ============================================================

class TestMSE:

    def test_perfect_prediction(self):
        """MSE should be 0 when prediction == cost."""
        d = 4
        costs = torch.randn(16, d)
        feats = costs.clone()  # identity model: pred = feats = costs
        sols = torch.randn(16, d)
        objs = torch.randn(16, 1)
        ds = TensorDataset(feats, costs, sols, objs)
        loader = DataLoader(ds, batch_size=8)

        model = _IdentityModel()
        mse = MSE(model, loader)
        assert abs(mse) < 1e-6

    def test_known_mse(self):
        """MSE with known offset."""
        d = 4
        n = 8
        costs = torch.zeros(n, d)
        feats = torch.ones(n, d)  # identity model predicts ones, true is zeros
        sols = torch.randn(n, d)
        objs = torch.randn(n, 1)
        ds = TensorDataset(feats, costs, sols, objs)
        loader = DataLoader(ds, batch_size=n)

        model = _IdentityModel()
        mse = MSE(model, loader)
        # MSE = mean over samples of mean-per-sample = mean of (d * 1^2 / d) = 1.0
        assert abs(mse - 1.0) < 1e-6

    def test_mse_non_negative(self):
        loader = _make_dataloader()
        model = _IdentityModel()
        mse = MSE(model, loader)
        assert mse >= 0

    def test_restores_train_mode(self):
        loader = _make_dataloader()
        model = _IdentityModel()
        model.train()
        MSE(model, loader)
        assert model.training


# ============================================================
# calRegret tests
# ============================================================

@requires_gurobi
class TestCalRegret:

    @pytest.fixture
    def sp_model(self):
        return shortestPathModel(grid=(3, 3))

    def test_zero_regret_with_true_cost(self, sp_model):
        """When pred_cost == true_cost, regret should be 0."""
        cost = np.random.RandomState(42).rand(sp_model.num_cost) + 0.1
        sp_model.setObj(cost)
        _, true_obj = sp_model.solve()
        loss = calRegret(sp_model, cost, cost, true_obj)
        assert abs(loss) < 1e-6

    def test_regret_non_negative_minimize(self, sp_model):
        """Regret should be non-negative for MINIMIZE."""
        rng = np.random.RandomState(42)
        cost_true = rng.rand(sp_model.num_cost) + 0.1
        cost_pred = rng.rand(sp_model.num_cost) + 0.1
        sp_model.setObj(cost_true)
        _, true_obj = sp_model.solve()
        loss = calRegret(sp_model, cost_pred, cost_true, true_obj)
        assert loss >= -1e-6

    def test_regret_non_negative_maximize(self):
        """Regret should be non-negative for MAXIMIZE."""
        weights = np.array([[3.0, 4.0, 5.0]])
        capacity = np.array([8.0])
        model = knapsackModel(weights=weights, capacity=capacity)
        rng = np.random.RandomState(42)
        cost_true = rng.rand(3) + 1.0
        cost_pred = rng.rand(3) + 1.0
        model.setObj(cost_true)
        _, true_obj = model.solve()
        loss = calRegret(model, cost_pred, cost_true, true_obj)
        assert loss >= -1e-6


# ============================================================
# SPOError tests
# ============================================================

@requires_gurobi
class TestSPOError:

    def test_perfect_prediction_zero_error(self):
        model = shortestPathModel(grid=(3, 3))
        model_type = type(model)
        args = {"grid": (3, 3)}
        rng = np.random.RandomState(42)
        costs = rng.rand(10, model.num_cost) + 0.1
        # perfect prediction
        error = SPOError(costs, costs, model_type, args)
        assert abs(error) < 1e-6

    def test_spo_error_non_negative(self):
        model = shortestPathModel(grid=(3, 3))
        model_type = type(model)
        args = {"grid": (3, 3)}
        rng = np.random.RandomState(42)
        true_cost = rng.rand(10, model.num_cost) + 0.1
        pred_cost = rng.rand(10, model.num_cost) + 0.1
        error = SPOError(pred_cost, true_cost, model_type, args)
        assert error >= -1e-6

    def test_shape_mismatch(self):
        model = shortestPathModel(grid=(3, 3))
        model_type = type(model)
        args = {"grid": (3, 3)}
        with pytest.raises(AssertionError):
            SPOError(np.ones((5, 12)), np.ones((5, 10)), model_type, args)


# ============================================================
# testMSE tests
# ============================================================

class TestTestMSE:

    def test_zero_error(self):
        costs = np.random.rand(10, 5)
        assert abs(_testMSE(costs, costs, None, None)) < 1e-10

    def test_known_value(self):
        pred = np.ones((4, 3))
        true = np.zeros((4, 3))
        assert abs(_testMSE(pred, true, None, None) - 1.0) < 1e-10

    def test_shape_mismatch(self):
        with pytest.raises(AssertionError):
            _testMSE(np.ones((5, 3)), np.ones((5, 4)), None, None)
