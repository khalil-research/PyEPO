#!/usr/bin/env python
"""Tests for pyepo.twostage: two-stage predict-then-optimize predictors.

sklearnPred wraps any single-output regressor into a multi-output one (no
solver). autoSklearnPred is Linux-only and heavy; only its graceful error
paths are checked when the package is absent.
"""

import numpy as np
import pytest

from pyepo.twostage import sklearnPred
from pyepo.twostage.autosklearnpred import _HAS_AUTO, autoSklearnPred

from .conftest import NUM_FEAT, requires_gurobi

requires_autosk = pytest.mark.skipif(not _HAS_AUTO, reason="auto-sklearn not installed")


class TestSklearnPred:
    def test_wraps_into_multioutput(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.multioutput import MultiOutputRegressor

        assert isinstance(sklearnPred(LinearRegression()), MultiOutputRegressor)

    def test_fit_and_predict_shape(self):
        from sklearn.linear_model import LinearRegression

        rng = np.random.RandomState(42)
        X, Y = rng.randn(50, 4), rng.randn(50, 3)
        est = sklearnPred(LinearRegression())
        est.fit(X, Y)
        assert est.predict(X[:10]).shape == (10, 3)

    def test_preserves_base_estimator(self):
        from sklearn.ensemble import RandomForestRegressor

        est = sklearnPred(RandomForestRegressor(n_estimators=3, random_state=0))
        assert isinstance(est.estimator, RandomForestRegressor)


@requires_gurobi
class TestSklearnPredPipeline:
    def test_end_to_end_regret(self):
        from sklearn.linear_model import LinearRegression

        import pyepo
        from pyepo.metric import SPOError
        from pyepo.model.grb.shortestpath import shortestPathModel

        x, c = pyepo.data.shortestpath.genData(40, NUM_FEAT, (3, 3), deg=1, seed=42)
        est = sklearnPred(LinearRegression())
        est.fit(x, c)
        c_pred = est.predict(x)
        assert c_pred.shape == c.shape
        err = SPOError(c_pred, c, shortestPathModel(grid=(3, 3)))
        assert err >= -1e-6


class TestAutoSklearnPred:
    def test_raises_when_missing(self):
        if _HAS_AUTO:
            pytest.skip("auto-sklearn is installed; skip negative test")
        with pytest.raises(ImportError):
            autoSklearnPred(optmodel=None, seed=0, timelimit=30, metric="mse")

    def test_mse_does_not_build_regret_scorer(self, monkeypatch):
        import pyepo.twostage.autosklearnpred as module

        captured = {}

        def fake_regressor(**kwargs):
            captured.update(kwargs)
            return kwargs

        monkeypatch.setattr(module, "_HAS_AUTO", True)
        monkeypatch.setattr(module, "AutoSklearnRegressor", fake_regressor, raising=False)
        monkeypatch.setattr(module, "mean_squared_error", object(), raising=False)
        monkeypatch.setattr(
            module,
            "makeAutoSkScorer",
            lambda _model: pytest.fail("MSE path should not build a regret scorer"),
        )

        result = module.autoSklearnPred(optmodel=None, seed=0, timelimit=30, metric="mse")

        assert result == captured
        assert captured["metric"] is module.mean_squared_error

    @requires_autosk
    def test_invalid_metric_raises(self):
        from pyepo.model.grb.shortestpath import shortestPathModel

        with pytest.raises(ValueError):
            autoSklearnPred(shortestPathModel(grid=(3, 3)), seed=0, timelimit=30, metric="banana")
