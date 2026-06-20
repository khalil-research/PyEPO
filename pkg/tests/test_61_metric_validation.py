"""Pure validation and helper tests for :mod:`pyepo.metric`."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyepo import EPO
from pyepo.metric._common import normalize_regret, validate_numpy_cost_batches
from pyepo.metric.metrics import SPOError
from pyepo.metric.regret import _regretFromObj, calRegret
from pyepo.metric.unambregret import calUnambRegret, unambRegret


class TestRegretFromObj:
    def test_minimize(self):
        assert _regretFromObj(3.0, 1.0, EPO.MINIMIZE) == 2.0

    def test_maximize(self):
        assert _regretFromObj(3.0, 5.0, EPO.MAXIMIZE) == 2.0

    def test_vectorized(self):
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
        validated_pred, validated_true = validate_numpy_cost_batches(
            pred, true, self.model.num_cost
        )
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
