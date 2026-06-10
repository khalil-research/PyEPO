#!/usr/bin/env python
"""Tests for pyepo.metric: MSE, regret, unambiguous regret, SPOError, scorers.

Per-sample helpers (calRegret, calUnambRegret, SPOError, testMSE) are checked
for the invariants that matter: perfect prediction => zero, regret >= 0 for both
senses, and unambiguous regret >= standard regret. The dataloader-level
regret() / unambRegret() / MSE() are run on a tiny untrained predictor to verify
they return sane floats. MSE / testMSE need no solver.
"""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pyepo.metric.metrics import SPOError, makeSkScorer
from pyepo.metric.metrics import testMSE as _testMSE
from pyepo.metric.mse import MSE
from pyepo.metric.regret import calRegret
from pyepo.metric.unambregret import calUnambRegret

from .conftest import NUM_FEAT, LinearPred, requires_gurobi


class _IdentityModel(nn.Module):
    """Prediction model that returns its input unchanged."""
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x


def _loader(n=16, d=4, batch_size=8):
    ds = TensorDataset(torch.randn(n, d), torch.randn(n, d), torch.randn(n, d), torch.randn(n, 1))
    return DataLoader(ds, batch_size=batch_size)


# ============================================================
# MSE / testMSE (no solver)
# ============================================================

class TestMSE:

    def test_perfect_prediction_zero(self):
        d = 4
        costs = torch.randn(16, d)
        ds = TensorDataset(costs.clone(), costs, torch.randn(16, d), torch.randn(16, 1))
        assert abs(MSE(_IdentityModel(), DataLoader(ds, batch_size=8))) < 1e-6

    def test_known_value(self):
        n, d = 8, 4
        ds = TensorDataset(torch.ones(n, d), torch.zeros(n, d), torch.randn(n, d), torch.randn(n, 1))
        # identity predicts ones, truth zeros => per-element error 1 => MSE 1.0
        assert abs(MSE(_IdentityModel(), DataLoader(ds, batch_size=n)) - 1.0) < 1e-6

    def test_non_negative(self):
        assert MSE(_IdentityModel(), _loader()) >= 0

    def test_restores_train_mode(self):
        m = _IdentityModel()
        m.train()
        MSE(m, _loader())
        assert m.training


class TestTestMSE:

    def test_zero_error(self):
        costs = np.random.rand(10, 5)
        assert abs(_testMSE(costs, costs, None, None)) < 1e-10

    def test_known_value(self):
        assert abs(_testMSE(np.ones((4, 3)), np.zeros((4, 3)), None, None) - 1.0) < 1e-10

    def test_shape_mismatch(self):
        with pytest.raises(AssertionError):
            _testMSE(np.ones((5, 3)), np.ones((5, 4)), None, None)


# ============================================================
# Per-sample regret helpers
# ============================================================

@requires_gurobi
class TestCalRegret:

    def _sp(self):
        from pyepo.model.grb.shortestpath import shortestPathModel
        return shortestPathModel(grid=(3, 3))

    def _ks(self):
        from pyepo.model.grb.knapsack import knapsackModel
        return knapsackModel(weights=np.array([[3.0, 4.0, 5.0]]), capacity=np.array([8.0]))

    def test_zero_with_true_cost(self):
        m = self._sp()
        cost = np.random.RandomState(42).rand(m.num_cost) + 0.1
        m.setObj(cost)
        _, true_obj = m.solve()
        assert abs(calRegret(m, cost, cost, true_obj)) < 1e-6

    def test_non_negative_minimize(self):
        m = self._sp()
        rng = np.random.RandomState(42)
        ct, cp = rng.rand(m.num_cost) + 0.1, rng.rand(m.num_cost) + 0.1
        m.setObj(ct)
        _, true_obj = m.solve()
        assert calRegret(m, cp, ct, true_obj) >= -1e-6

    def test_non_negative_maximize(self):
        m = self._ks()
        rng = np.random.RandomState(42)
        ct, cp = rng.rand(3) + 1.0, rng.rand(3) + 1.0
        m.setObj(ct)
        _, true_obj = m.solve()
        assert calRegret(m, cp, ct, true_obj) >= -1e-6


@requires_gurobi
class TestSPOError:

    def _sp(self):
        from pyepo.model.grb.shortestpath import shortestPathModel
        return shortestPathModel(grid=(3, 3))

    def test_perfect_prediction_zero(self):
        m = self._sp()
        costs = np.random.RandomState(42).rand(10, m.num_cost) + 0.1
        assert abs(SPOError(costs, costs, type(m), {"grid": (3, 3)})) < 1e-6

    def test_non_negative(self):
        m = self._sp()
        rng = np.random.RandomState(42)
        true_c = rng.rand(10, m.num_cost) + 0.1
        pred_c = rng.rand(10, m.num_cost) + 0.1
        assert SPOError(pred_c, true_c, type(m), {"grid": (3, 3)}) >= -1e-6

    def test_non_negative_maximize(self):
        from pyepo.model.grb.knapsack import knapsackModel
        weights, cap = np.array([[3.0, 4.0, 5.0]]), np.array([8.0])
        m = knapsackModel(weights=weights, capacity=cap)
        rng = np.random.RandomState(42)
        true_c = rng.rand(10, m.num_cost) + 1.0
        pred_c = rng.rand(10, m.num_cost) + 1.0
        assert SPOError(pred_c, true_c, type(m), {"weights": weights, "capacity": cap}) >= -1e-6

    def test_shape_mismatch(self):
        m = self._sp()
        with pytest.raises(AssertionError):
            SPOError(np.ones((5, 12)), np.ones((5, 10)), type(m), {"grid": (3, 3)})


@requires_gurobi
class TestCalUnambRegret:

    def _sp(self):
        from pyepo.model.grb.shortestpath import shortestPathModel
        return shortestPathModel(grid=(3, 3))

    def test_zero_with_true_cost(self):
        m = self._sp()
        cost = np.random.RandomState(42).rand(m.num_cost) + 0.1
        m.setObj(cost)
        _, true_obj = m.solve()
        assert abs(calUnambRegret(m, cost, cost, true_obj)) < 1e-3

    def test_non_negative(self):
        m = self._sp()
        rng = np.random.RandomState(42)
        ct, cp = rng.rand(m.num_cost) + 0.1, rng.rand(m.num_cost) + 0.1
        m.setObj(ct)
        _, true_obj = m.solve()
        assert calUnambRegret(m, cp, ct, true_obj) >= -1e-3

    def test_non_negative_maximize(self):
        from pyepo.model.grb.knapsack import knapsackModel
        m = knapsackModel(weights=np.array([[3.0, 4.0, 5.0]]), capacity=np.array([8.0]))
        rng = np.random.RandomState(42)
        ct, cp = rng.rand(m.num_cost) + 1.0, rng.rand(m.num_cost) + 1.0
        m.setObj(ct)
        _, true_obj = m.solve()
        assert calUnambRegret(m, cp, ct, true_obj) >= -1e-3

    def test_at_least_standard_regret(self):
        m = self._sp()
        rng = np.random.RandomState(7)
        ct, cp = rng.rand(m.num_cost) + 0.1, rng.rand(m.num_cost) + 0.1
        m.setObj(ct)
        _, true_obj = m.solve()
        std = calRegret(m, cp, ct, true_obj)
        unamb = calUnambRegret(m, cp, ct, true_obj)
        assert unamb >= std - 1e-3

    def test_continuous_maximize_fractional_optimum(self):
        from pyepo import dsl

        x = dsl.Variable(2, lb=0, ub=1)
        c = dsl.Parameter(2)
        m = dsl.Problem(dsl.Maximize(c @ x), [np.array([1.0, 3.0]) @ x <= 1.0]).compile("gurobi")
        # fractional predicted optimum (x2 = 1/3)
        unamb = calUnambRegret(m, np.array([1.0, 4.0]), np.array([1.0, 1.0]), true_obj=1.0)
        assert unamb == pytest.approx(2.0 / 3.0, abs=1e-3)

    def test_partial_prediction_scales_fixed_cost(self):
        from pyepo import dsl

        x = dsl.Variable(1, lb=0, ub=1)
        y = dsl.Variable(1, lb=0, ub=1)
        c = dsl.Parameter(1)
        m = dsl.Problem(
            dsl.Minimize(c @ x + np.array([5.0]) @ y), [x.sum() + y.sum() >= 1]
        ).compile("gurobi")
        # cp = 2 < 5: the tie set is {x}, not the fixed-cost var y
        unamb = calUnambRegret(m, np.array([2.0]), np.array([3.0]), true_obj=3.0)
        assert unamb == pytest.approx(0.0, abs=1e-3)

    def test_max_iter_raises(self):
        m = self._sp()
        cost = np.random.RandomState(42).rand(m.num_cost) + 0.1
        m.setObj(cost)
        _, true_obj = m.solve()
        with pytest.raises(RuntimeError):
            calUnambRegret(m, cost, cost, true_obj, max_iter=0)

    def test_worst_case_includes_offset(self):
        # tie {[1,0],[0,1]} from full pred [cp+d]=[0,0]; true full [10,5], z*=5, worst=10 -> regret 5
        from pyepo import EPO, dsl

        x = dsl.Variable(2, vtype=EPO.BINARY)
        c = dsl.Parameter(2)
        d = np.array([10.0, 0.0])
        m = dsl.Problem(dsl.Minimize((c + d) @ x), [x.sum() == 1]).compile(backend="gurobi")
        loss = calUnambRegret(
            m, np.array([-10.0, 0.0]), np.array([0.0, 5.0]), true_obj=5.0, tolerance=1.0
        )
        assert loss == pytest.approx(5.0)


# ============================================================
# Dataloader-level metrics (untrained predictor -> sane floats)
# ============================================================

@requires_gurobi
class TestDataloaderMetrics:

    def test_regret_minimize(self, sp_data):
        import pyepo
        optmodel, _ds, loader = sp_data
        reg = pyepo.metric.regret(LinearPred(NUM_FEAT, optmodel.num_cost), optmodel, loader)
        assert isinstance(reg, float) and reg >= 0

    def test_regret_maximize(self, ks_data):
        import pyepo
        optmodel, _ds, loader = ks_data
        reg = pyepo.metric.regret(LinearPred(NUM_FEAT, optmodel.num_cost), optmodel, loader)
        assert isinstance(reg, float) and reg >= 0

    def test_mse(self, sp_data):
        import pyepo
        optmodel, _ds, loader = sp_data
        mse = pyepo.metric.MSE(LinearPred(NUM_FEAT, optmodel.num_cost), loader)
        assert isinstance(mse, float) and mse >= 0

    def test_unamb_regret(self, sp_data):
        import pyepo
        optmodel, _ds, loader = sp_data
        unamb = pyepo.metric.unambRegret(LinearPred(NUM_FEAT, optmodel.num_cost), optmodel, loader)
        assert isinstance(unamb, float) and unamb >= -1e-6


# ============================================================
# sklearn scorer
# ============================================================

@requires_gurobi
class TestSkScorer:

    def test_scorer_returns_finite_float(self):
        from sklearn.linear_model import LinearRegression

        import pyepo
        from pyepo.model.grb.shortestpath import shortestPathModel
        from pyepo.twostage import sklearnPred
        x, c = pyepo.data.shortestpath.genData(40, NUM_FEAT, (3, 3), seed=42)
        optmodel = shortestPathModel(grid=(3, 3))
        est = sklearnPred(LinearRegression())
        est.fit(x, c)
        scorer = makeSkScorer(optmodel)
        score = scorer(est, x, c)
        # greater_is_better=False => scorer returns negated regret (<= 0)
        assert np.isfinite(score)
        assert score <= 1e-6

    def test_scorer_argument_orientation(self):
        from sklearn.linear_model import LinearRegression

        import pyepo
        from pyepo.model.grb.shortestpath import shortestPathModel
        from pyepo.twostage import sklearnPred
        from pyepo.utils import getArgs

        # nonlinear costs + noise keep the linear fit imperfect, so regret is asymmetric
        x, c = pyepo.data.shortestpath.genData(20, NUM_FEAT, (3, 3), deg=4, noise_width=0.5, seed=42)
        optmodel = shortestPathModel(grid=(3, 3))
        est = sklearnPred(LinearRegression())
        est.fit(x, c)
        cp = est.predict(x)
        scorer = makeSkScorer(optmodel)
        # sklearn calls score_func(y_true, y_pred): the scorer must negate SPOError(pred, true)
        expected = -SPOError(cp, c, type(optmodel), getArgs(optmodel))
        swapped = -SPOError(c, cp, type(optmodel), getArgs(optmodel))
        assert expected < -1e-3
        assert np.isclose(scorer(est, x, c), expected)
        # the orientation is observable: swapping the arguments changes the value
        assert not np.isclose(expected, swapped)


class TestAutoSkScorer:

    def test_raises_without_autosklearn(self):
        from pyepo.metric.metrics import makeAutoSkScorer
        from pyepo.twostage.autosklearnpred import _HAS_AUTO

        if _HAS_AUTO:
            pytest.skip("auto-sklearn is installed; skip negative test")
        # the lazy autosklearn import fails before the model is touched
        with pytest.raises(ImportError):
            makeAutoSkScorer(None)
