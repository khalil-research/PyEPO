#!/usr/bin/env python
"""Tests for pyepo.metric: MSE, regret, unambiguous regret, SPOError, scorers.

Per-sample helpers (calRegret, calUnambRegret, SPOError) are checked
for the invariants that matter: perfect prediction => zero, regret >= 0 for both
senses, and unambiguous regret >= standard regret. The dataloader-level
regret() / unambRegret() / MSE() are run on a tiny untrained predictor to verify
they return sane floats. MSE needs no solver.
"""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pyepo.metric._common import normalize_regret
from pyepo.metric.metrics import SPOError, _validate_cost_batches, makeSkScorer
from pyepo.metric.mse import MSE
from pyepo.metric.regret import _regretFromObj, calRegret, regret
from pyepo.metric.unambregret import calUnambRegret, unambRegret

from .conftest import (
    _HAS_FLAX,
    _HAS_GUROBI,
    BATCH,
    GRID,
    NUM_DATA,
    NUM_FEAT,
    LinearPred,
    requires_gurobi,
    requires_mpax,
)


class _IdentityModel(nn.Module):
    """Prediction model that returns its input unchanged."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x


class _NoParamModel(nn.Module):
    """Parameterless predictor that returns its input unchanged."""

    def forward(self, x):
        return x


class _FailingModel(nn.Module):
    """Predictor used to verify evaluation cleanup after an exception."""

    def forward(self, x):
        raise RuntimeError("prediction failed")


class _WrongWidthModel(nn.Module):
    """Predictor that drops one cost from every sample."""

    def forward(self, x):
        return x[:, :-1]


class _NonfiniteModel(nn.Module):
    """Predictor that returns a non-finite cost."""

    def forward(self, x):
        output = x.clone()
        output[0, 0] = torch.nan
        return output


def _loader(n=16, d=4, batch_size=8):
    ds = TensorDataset(torch.randn(n, d), torch.randn(n, d), torch.randn(n, d), torch.randn(n, 1))
    return DataLoader(ds, batch_size=batch_size)


# ============================================================
# MSE (no solver)
# ============================================================


class TestMSE:
    def test_perfect_prediction_zero(self):
        d = 4
        costs = torch.randn(16, d)
        ds = TensorDataset(costs.clone(), costs, torch.randn(16, d), torch.randn(16, 1))
        assert abs(MSE(_IdentityModel(), DataLoader(ds, batch_size=8))) < 1e-6

    def test_known_value(self):
        n, d = 8, 4
        ds = TensorDataset(
            torch.ones(n, d), torch.zeros(n, d), torch.randn(n, d), torch.randn(n, 1)
        )
        # identity predicts ones, truth zeros => per-element error 1 => MSE 1.0
        assert abs(MSE(_IdentityModel(), DataLoader(ds, batch_size=n)) - 1.0) < 1e-6

    def test_non_negative(self):
        assert MSE(_IdentityModel(), _loader()) >= 0

    def test_restores_train_mode(self):
        m = _IdentityModel()
        m.train()
        MSE(m, _loader())
        assert m.training

    def test_preserves_eval_mode(self):
        m = _IdentityModel()
        m.eval()
        MSE(m, _loader())
        assert not m.training

    def test_parameterless_model(self):
        assert MSE(_NoParamModel(), _loader()) >= 0

    @pytest.mark.parametrize("training", [True, False])
    def test_restores_mode_after_prediction_error(self, training):
        model = _FailingModel()
        model.train(training)
        with pytest.raises(RuntimeError, match="prediction failed"):
            MSE(model, _loader())
        assert model.training is training


class TestDataloaderPredictionValidation:
    optmodel = SimpleNamespace(num_cost=4)

    def test_mse_rejects_wrong_prediction_shape(self):
        model = _WrongWidthModel()
        with pytest.raises(ValueError, match="does not match"):
            MSE(model, _loader())
        assert model.training

    def test_mse_rejects_nonfinite_prediction(self):
        with pytest.raises(ValueError, match="finite"):
            MSE(_NonfiniteModel(), _loader())

    @pytest.mark.parametrize("metric", [regret, unambRegret])
    def test_regret_metrics_reject_wrong_prediction_width_before_solver(self, metric):
        model = _WrongWidthModel()
        with pytest.raises(ValueError, match="does not match"):
            metric(model, self.optmodel, _loader())
        assert model.training

    @pytest.mark.parametrize("metric", [regret, unambRegret])
    def test_regret_metrics_reject_nonfinite_prediction_before_solver(self, metric):
        with pytest.raises(ValueError, match="finite"):
            metric(_NonfiniteModel(), self.optmodel, _loader())

    def test_callable_regret_rejects_wrong_prediction_width_before_solver(self, monkeypatch):
        def predict(x):
            return np.ones((x.shape[0], 3), dtype=np.float32)

        def fail_pool_creation(*args, **kwargs):
            pytest.fail("solver pool initialized before prediction validation")

        regret_module = importlib.import_module("pyepo.metric.regret")
        monkeypatch.setattr(regret_module, "create_solver_pool", fail_pool_creation)
        with pytest.raises(ValueError, match="does not match"):
            regret(predict, self.optmodel, _loader())

    @pytest.mark.parametrize("metric", [MSE, regret, unambRegret])
    def test_rejects_complex_prediction(self, metric):
        class ComplexModel(nn.Module):
            def forward(self, x):
                return x.to(torch.complex64)

        args = (ComplexModel(), _loader())
        if metric is not MSE:
            args = (ComplexModel(), self.optmodel, _loader())
        with pytest.raises(ValueError, match="numerical"):
            metric(*args)


class TestRegretFromObj:
    def test_minimize(self):
        from pyepo import EPO

        assert _regretFromObj(3.0, 1.0, EPO.MINIMIZE) == 2.0

    def test_maximize(self):
        from pyepo import EPO

        assert _regretFromObj(3.0, 5.0, EPO.MAXIMIZE) == 2.0

    def test_vectorized(self):
        from pyepo import EPO

        out = _regretFromObj(np.array([3.0, 4.0]), np.array([1.0, 1.0]), EPO.MINIMIZE)
        np.testing.assert_allclose(out, [2.0, 3.0])

    def test_invalid_sense(self):
        with pytest.raises(ValueError):
            _regretFromObj(1.0, 1.0, "bad")


class TestNormalizeRegret:
    def test_normalizes_by_absolute_optimum_sum(self):
        assert normalize_regret(6.0, 3.0) == pytest.approx(2.0)

    def test_zero_regret_remains_zero_with_zero_optimum_sum(self):
        assert normalize_regret(0.0, 0.0) == 0.0

    def test_zero_optimum_sum_uses_defensive_epsilon(self):
        assert normalize_regret(1.0, 0.0) == pytest.approx(1e8)


class TestSPOErrorValidation:
    model = SimpleNamespace(num_cost=4)

    def test_validation_preserves_cost_dtype(self):
        pred = np.ones((2, 4), dtype=np.float32)
        true = np.ones((2, 4), dtype=np.float32)
        validated_pred, validated_true = _validate_cost_batches(pred, true, self.model.num_cost)
        assert validated_pred.dtype == np.float32
        assert validated_true.dtype == np.float32

    def test_rejects_shape_mismatch_before_solver_access(self):
        with pytest.raises(ValueError, match="does not match"):
            SPOError(np.ones((2, 4)), np.ones((3, 4)), self.model)

    @pytest.mark.parametrize(
        ("pred_cost", "true_cost"),
        [
            (np.ones(4), np.ones(4)),
            (np.ones((1, 2, 4)), np.ones((1, 2, 4))),
        ],
    )
    def test_rejects_non_batch_inputs_before_solver_access(self, pred_cost, true_cost):
        with pytest.raises(ValueError, match="two-dimensional"):
            SPOError(pred_cost, true_cost, self.model)

    def test_rejects_empty_batch_before_solver_access(self):
        with pytest.raises(ValueError, match="must not be empty"):
            SPOError(np.empty((0, 4)), np.empty((0, 4)), self.model)

    def test_rejects_wrong_cost_width_before_solver_access(self):
        with pytest.raises(ValueError, match="num_cost"):
            SPOError(np.ones((2, 3)), np.ones((2, 3)), self.model)

    @pytest.mark.parametrize("cost_kind", ["predicted", "true"])
    @pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
    def test_rejects_nonfinite_cost_before_solver_access(self, cost_kind, invalid):
        pred = np.ones((2, 4))
        true = np.ones((2, 4))
        target = pred if cost_kind == "predicted" else true
        target[0, 0] = invalid
        with pytest.raises(ValueError, match="finite"):
            SPOError(pred, true, self.model)

    def test_rejects_nonnumeric_cost_before_solver_access(self):
        with pytest.raises(ValueError, match="numerical"):
            SPOError([["bad"] * 4], [["bad"] * 4], self.model)


class TestSingleRegretValidation:
    @staticmethod
    def _call(metric, pred_cost, true_cost, true_obj=0.0):
        metric(SimpleNamespace(num_cost=4), pred_cost, true_cost, true_obj)

    @pytest.mark.parametrize("metric", [calRegret, calUnambRegret])
    @pytest.mark.parametrize(
        ("pred_cost", "true_cost"),
        [
            (np.ones((1, 4)), np.ones((1, 4))),
            (np.ones(4), np.ones((1, 4))),
        ],
    )
    def test_rejects_nonvector_cost_before_solver_access(self, metric, pred_cost, true_cost):
        with pytest.raises(ValueError, match="one-dimensional"):
            self._call(metric, pred_cost, true_cost)

    @pytest.mark.parametrize("metric", [calRegret, calUnambRegret])
    def test_rejects_mismatched_cost_shapes_before_solver_access(self, metric):
        with pytest.raises(ValueError, match="does not match"):
            self._call(metric, np.ones(4), np.ones(3))

    @pytest.mark.parametrize("metric", [calRegret, calUnambRegret])
    def test_rejects_wrong_cost_length_before_solver_access(self, metric):
        with pytest.raises(ValueError, match="num_cost"):
            self._call(metric, np.ones(3), np.ones(3))

    @pytest.mark.parametrize("metric", [calRegret, calUnambRegret])
    @pytest.mark.parametrize("cost_kind", ["predicted", "true"])
    @pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
    def test_rejects_nonfinite_cost_before_solver_access(self, metric, cost_kind, invalid):
        pred = np.ones(4)
        true = np.ones(4)
        target = pred if cost_kind == "predicted" else true
        target[0] = invalid
        with pytest.raises(ValueError, match="finite"):
            self._call(metric, pred, true)

    @pytest.mark.parametrize("metric", [calRegret, calUnambRegret])
    def test_rejects_nonnumeric_cost_before_solver_access(self, metric):
        with pytest.raises(ValueError, match="numerical"):
            self._call(metric, ["bad"] * 4, ["bad"] * 4)

    @pytest.mark.parametrize("metric", [calRegret, calUnambRegret])
    @pytest.mark.parametrize("true_obj", [np.nan, np.inf, -np.inf, True, "bad"])
    def test_rejects_invalid_true_objective_before_solver_access(self, metric, true_obj):
        with pytest.raises(ValueError, match="true_obj"):
            self._call(metric, np.ones(4), np.ones(4), true_obj)


class TestUnambRegretValidation:
    def test_public_metric_rejects_invalid_tolerance_before_model_access(self):
        with pytest.raises(ValueError, match="tolerance"):
            unambRegret(None, None, None, tolerance=0.0)

    def test_public_metric_rejects_invalid_retry_count_before_model_access(self):
        with pytest.raises(ValueError, match="max_iter"):
            unambRegret(None, None, None, max_iter=True)

    @pytest.mark.parametrize("tolerance", [0.0, -1.0, np.nan, np.inf, True])
    def test_rejects_invalid_tolerance_before_model_access(self, tolerance):
        with pytest.raises(ValueError, match="tolerance"):
            calUnambRegret(None, np.ones(2), np.ones(2), 0.0, tolerance=tolerance)

    @pytest.mark.parametrize("max_iter", [1.5, True])
    def test_rejects_noninteger_retry_count_before_model_access(self, max_iter):
        with pytest.raises(ValueError, match="max_iter"):
            calUnambRegret(None, np.ones(2), np.ones(2), 0.0, max_iter=max_iter)

    @pytest.mark.parametrize("max_iter", [0, -1])
    def test_exhausted_retry_budget_preserves_runtime_error(self, max_iter):
        with pytest.raises(RuntimeError, match="Max iterations"):
            calUnambRegret(None, np.ones(2), np.ones(2), 0.0, max_iter=max_iter)


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

    def test_zero_with_objective_constant(self):
        from pyepo import EPO, dsl

        x = dsl.Variable(2, vtype=EPO.BINARY)
        c = dsl.Parameter(2)
        # (x - 1).sum() adds +1 per var to fixed_cost and obj_offset -2
        m = dsl.Problem(dsl.Minimize(c @ x + (x - 1).sum()), [x.sum() >= 1]).compile(
            backend="gurobi"
        )
        cost = np.array([1.0, 2.0])
        m.setObj(cost)
        _, true_obj = m.solve()
        assert abs(calRegret(m, cost, cost, true_obj)) < 1e-6

    def test_rejects_quadratic_objective(self):
        from pyepo import dsl

        x = dsl.Variable(2, lb=0, ub=1)
        c = dsl.Parameter(2)
        m = dsl.Problem(dsl.Minimize(c @ x + (x - 1) @ (x - 1)), [x.sum() >= 0.0]).compile(
            backend="gurobi"
        )
        cost = np.array([1.0, -1.0])
        with pytest.raises(ValueError):
            calRegret(m, cost, cost, 0.0)
        with pytest.raises(ValueError):
            calUnambRegret(m, cost, cost, 0.0)
        with pytest.raises(ValueError):
            SPOError(cost.reshape(1, -1), cost.reshape(1, -1), m)
        import pyepo

        with pytest.raises(ValueError):
            pyepo.metric.regret(_IdentityModel(), m, _loader())
        with pytest.raises(ValueError):
            pyepo.metric.unambRegret(_IdentityModel(), m, _loader())


@requires_gurobi
class TestSPOError:
    def _sp(self):
        from pyepo.model.grb.shortestpath import shortestPathModel

        return shortestPathModel(grid=(3, 3))

    def test_perfect_prediction_zero(self):
        m = self._sp()
        costs = np.random.RandomState(42).rand(10, m.num_cost) + 0.1
        assert abs(SPOError(costs, costs, m)) < 1e-6

    def test_non_negative(self):
        m = self._sp()
        rng = np.random.RandomState(42)
        true_c = rng.rand(10, m.num_cost) + 0.1
        pred_c = rng.rand(10, m.num_cost) + 0.1
        assert SPOError(pred_c, true_c, m) >= -1e-6

    def test_non_negative_maximize(self):
        from pyepo.model.grb.knapsack import knapsackModel

        weights, cap = np.array([[3.0, 4.0, 5.0]]), np.array([8.0])
        m = knapsackModel(weights=weights, capacity=cap)
        rng = np.random.RandomState(42)
        true_c = rng.rand(10, m.num_cost) + 1.0
        pred_c = rng.rand(10, m.num_cost) + 1.0
        assert SPOError(pred_c, true_c, m) >= -1e-6


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

    def test_zero_with_objective_constant(self):
        from pyepo import EPO, dsl

        x = dsl.Variable(2, vtype=EPO.BINARY)
        c = dsl.Parameter(2)
        m = dsl.Problem(dsl.Minimize(c @ x + (x - 1).sum()), [x.sum() >= 1]).compile(
            backend="gurobi"
        )
        cost = np.array([1.0, 2.0])
        m.setObj(cost)
        _, true_obj = m.solve()
        assert abs(calUnambRegret(m, cost, cost, true_obj, tolerance=1.0)) < 1e-3


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
        pred = LinearPred(NUM_FEAT, optmodel.num_cost)
        pred.eval()
        unamb = pyepo.metric.unambRegret(pred, optmodel, loader)
        assert isinstance(unamb, float) and unamb >= -1e-6
        assert not pred.training

    def test_regret_preserves_eval_mode(self, sp_data):
        import pyepo

        optmodel, _ds, loader = sp_data
        pred = LinearPred(NUM_FEAT, optmodel.num_cost)
        pred.eval()
        pyepo.metric.regret(pred, optmodel, loader)
        assert not pred.training

    def test_regret_reductions_consistent(self, sp_data):
        import pyepo

        optmodel, ds, loader = sp_data
        torch.manual_seed(42)
        pred = LinearPred(NUM_FEAT, optmodel.num_cost)
        per = pyepo.metric.regret(pred, optmodel, loader, reduction="none")
        total = pyepo.metric.regret(pred, optmodel, loader, reduction="sum")
        mean = pyepo.metric.regret(pred, optmodel, loader, reduction="mean")
        norm = pyepo.metric.regret(pred, optmodel, loader)
        assert isinstance(per, np.ndarray) and len(per) == len(ds)
        assert total == pytest.approx(per.sum(), rel=1e-5)
        assert mean == pytest.approx(per.mean(), rel=1e-5)
        zsum = sum(abs(z).sum().item() for _, _, _, z in loader)
        assert norm == pytest.approx(total / zsum, rel=1e-4)

    def test_regret_invalid_reduction(self, sp_data):
        import pyepo

        optmodel, _ds, loader = sp_data
        with pytest.raises(ValueError):
            pyepo.metric.regret(
                LinearPred(NUM_FEAT, optmodel.num_cost), optmodel, loader, reduction="bad"
            )

    def test_regret_multiprocess_matches_single(self, sp_data):
        import pyepo

        optmodel, _ds, loader = sp_data
        torch.manual_seed(0)
        pred = LinearPred(NUM_FEAT, optmodel.num_cost)
        single = pyepo.metric.regret(pred, optmodel, loader)
        multi = pyepo.metric.regret(pred, optmodel, loader, processes=2)
        assert multi == pytest.approx(single, rel=1e-5)

    def test_unamb_regret_max_iter_passthrough(self, sp_data):
        import pyepo

        optmodel, _ds, loader = sp_data
        with pytest.raises(RuntimeError):
            pyepo.metric.unambRegret(
                LinearPred(NUM_FEAT, optmodel.num_cost), optmodel, loader, max_iter=0
            )

    def test_regret_offset_dataloader(self):
        import pyepo
        from pyepo import EPO, dsl
        from pyepo.data.dataset import optDataset

        x = dsl.Variable(2, vtype=EPO.BINARY)
        c = dsl.Parameter(2)
        m = dsl.Problem(dsl.Minimize(c @ x + (x - 1).sum()), [x.sum() >= 1]).compile(
            backend="gurobi"
        )
        costs = (np.random.RandomState(42).rand(6, 2) + 0.1).astype(np.float32)
        ds = optDataset(m, costs, costs)
        loader = DataLoader(ds, batch_size=3)
        # identity predictor makes predictions perfect, so regret must vanish
        assert abs(pyepo.metric.regret(_IdentityModel(), m, loader)) < 1e-5


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

        # nonlinear costs + noise keep the linear fit imperfect, so regret is asymmetric
        x, c = pyepo.data.shortestpath.genData(
            20, NUM_FEAT, (3, 3), deg=4, noise_width=0.5, seed=42
        )
        optmodel = shortestPathModel(grid=(3, 3))
        est = sklearnPred(LinearRegression())
        est.fit(x, c)
        cp = est.predict(x)
        scorer = makeSkScorer(optmodel)
        # sklearn calls score_func(y_true, y_pred): the scorer must negate SPOError(pred, true)
        expected = -SPOError(cp, c, optmodel)
        swapped = -SPOError(c, cp, optmodel)
        assert expected < -1e-3
        assert np.isclose(scorer(est, x, c), expected)
        # the orientation is observable: swapping the arguments changes the value
        assert not np.isclose(expected, swapped)


# ============================================================
# JAX callable path for pyepo.metric.regret
# ============================================================


@requires_mpax
class TestDataloaderMetricsJax:
    """pyepo.metric.regret accepts a plain callable f(x_numpy) -> array."""

    @pytest.fixture(scope="class")
    def mpax_data(self):
        import pyepo
        from pyepo.data.dataset import optDataset
        from pyepo.model.mpax.shortestpath import shortestPathModel

        x, c = pyepo.data.shortestpath.genData(NUM_DATA, NUM_FEAT, GRID, seed=42)
        optmodel = shortestPathModel(grid=GRID)
        dataset = optDataset(optmodel, x, c)
        loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)
        return optmodel, dataset, loader

    def test_callable_returns_non_negative_float(self, mpax_data):
        import pyepo

        optmodel, _ds, loader = mpax_data
        fn = lambda x: np.ones((x.shape[0], optmodel.num_cost), dtype=np.float32)  # noqa: E731
        reg = pyepo.metric.regret(fn, optmodel, loader)
        assert isinstance(reg, float) and reg >= 0

    def test_callable_reductions_consistent(self, mpax_data):
        import pyepo

        optmodel, ds, loader = mpax_data
        fn = lambda x: np.ones((x.shape[0], optmodel.num_cost), dtype=np.float32)  # noqa: E731
        per = pyepo.metric.regret(fn, optmodel, loader, reduction="none")
        total = pyepo.metric.regret(fn, optmodel, loader, reduction="sum")
        assert isinstance(per, np.ndarray) and len(per) == len(ds)
        assert total == pytest.approx(per.sum(), rel=1e-5)


@pytest.mark.skipif(
    not (_HAS_GUROBI and _HAS_FLAX),
    reason="Parity: Gurobi (exact solve) + Flax both required",
)
class TestDataloaderMetricsJaxParity:
    """JAX callable and torch nn.Module with identical weights give the same regret."""

    def test_same_weights_same_regret(self, sp_data):
        import functools

        import jax.numpy as jnp
        from flax import linen as nn

        import pyepo

        optmodel, _ds, loader = sp_data

        # random torch predictor; capture weights as numpy
        torch_pred = LinearPred(NUM_FEAT, optmodel.num_cost)
        torch_pred.eval()
        w = torch_pred.linear.weight.detach().numpy()  # (num_cost, num_feat)
        b = torch_pred.linear.bias.detach().numpy()  # (num_cost,)

        # Flax Dense: kernel shape is (num_feat, num_cost) = w.T
        flax_pred = nn.Dense(optmodel.num_cost)
        params = {"params": {"kernel": jnp.asarray(w.T), "bias": jnp.asarray(b)}}

        torch_reg = pyepo.metric.regret(torch_pred, optmodel, loader)
        jax_reg = pyepo.metric.regret(functools.partial(flax_pred.apply, params), optmodel, loader)
        assert jax_reg == pytest.approx(torch_reg, abs=1e-4)


class TestAutoSkScorer:
    def test_raises_without_autosklearn(self):
        from pyepo.metric.metrics import makeAutoSkScorer
        from pyepo.twostage.autosklearnpred import _HAS_AUTO

        if _HAS_AUTO:
            pytest.skip("auto-sklearn is installed; skip negative test")
        # the lazy autosklearn import fails before the model is touched
        with pytest.raises(ImportError):
            makeAutoSkScorer(None)
