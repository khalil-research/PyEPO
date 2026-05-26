#!/usr/bin/env python
"""
Tests for pyepo.twostage: two-stage predict-then-optimize predictors
"""

import numpy as np
import pytest

from pyepo.twostage import sklearnPred
from pyepo.twostage.autosklearnpred import _HAS_AUTO, autoSklearnPred

try:
    from pyepo.model.grb.shortestpath import shortestPathModel
    _HAS_GUROBI = True
except (ImportError, NameError):
    _HAS_GUROBI = False

requires_gurobi = pytest.mark.skipif(not _HAS_GUROBI, reason="Gurobi not installed")
requires_autosk = pytest.mark.skipif(not _HAS_AUTO, reason="auto-sklearn not installed")


# ============================================================
# sklearnPred tests
# ============================================================

class TestSklearnPred:

    def test_wraps_regressor(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.multioutput import MultiOutputRegressor
        wrapped = sklearnPred(LinearRegression())
        assert isinstance(wrapped, MultiOutputRegressor)

    def test_fit_and_predict_shape(self):
        from sklearn.linear_model import LinearRegression
        rng = np.random.RandomState(42)
        X = rng.randn(50, 4)
        # multi-output target: 3 cost components
        Y = rng.randn(50, 3)
        wrapped = sklearnPred(LinearRegression())
        wrapped.fit(X, Y)
        pred = wrapped.predict(X[:10])
        assert pred.shape == (10, 3)

    def test_preserves_estimator_type(self):
        """sklearnPred should accept any sklearn regressor."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        wrapped = sklearnPred(RandomForestRegressor(n_estimators=3, random_state=0))
        assert isinstance(wrapped, MultiOutputRegressor)
        # the base estimator is the one passed in
        assert isinstance(wrapped.estimator, RandomForestRegressor)


# ============================================================
# sklearnPred integration with PyEPO pipeline
# ============================================================

@requires_gurobi
class TestSklearnPredIntegration:

    def test_end_to_end_pipeline(self):
        """Fit a two-stage model on synthetic data and evaluate regret."""
        from sklearn.linear_model import LinearRegression
        from torch.utils.data import DataLoader

        import pyepo
        from pyepo.data.dataset import optDataset
        # generate small dataset
        x, c = pyepo.data.shortestpath.genData(
            num_data=40, num_features=3, grid=(3, 3), deg=1, seed=42)
        optmodel = shortestPathModel(grid=(3, 3))
        dataset = optDataset(optmodel, x, c)
        _ = DataLoader(dataset, batch_size=8, shuffle=False)
        # train two-stage predictor
        predictor = sklearnPred(LinearRegression())
        predictor.fit(x, c)
        # predict
        c_pred = predictor.predict(x)
        assert c_pred.shape == c.shape
        # evaluate regret using the underlying pipeline
        from pyepo.metric import SPOError
        err = SPOError(c_pred, c, shortestPathModel, {"grid": (3, 3)})
        assert err >= -1e-6


# ============================================================
# autoSklearnPred tests (graceful skip when auto-sklearn missing)
# ============================================================

class TestAutoSklearnPred:

    def test_raises_when_autosklearn_missing(self):
        """When auto-sklearn is not installed, calling should raise ImportError."""
        if _HAS_AUTO:
            pytest.skip("auto-sklearn is installed; skip negative test")
        with pytest.raises(ImportError):
            autoSklearnPred(optmodel=None, seed=0, timelimit=30, metric="mse")

    @requires_autosk
    def test_invalid_metric_raises(self):
        """Invalid metric string should raise ValueError."""
        from pyepo.model.grb.shortestpath import shortestPathModel
        optmodel = shortestPathModel(grid=(3, 3))
        with pytest.raises(ValueError):
            autoSklearnPred(optmodel, seed=0, timelimit=30, metric="banana")
