#!/usr/bin/env python
"""
Two-stage model with Scikit-learn predictor
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.multioutput import MultiOutputRegressor

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


def sklearnPred(pmodel: BaseEstimator) -> MultiOutputRegressor:
    """
    Wrap a scikit-learn estimator into a multi-output regressor for two-stage baselines.

    The two-stage approach trains a regression model to minimize prediction
    error (e.g. MSE on cost coefficients) and only afterwards plugs the
    predicted costs into the optimization solver. This helper turns any
    single-output scikit-learn estimator (``LinearRegression``,
    ``RandomForestRegressor``, ``MLPRegressor``, ...) into a multi-output
    regressor suitable for predicting the full cost vector.

    Args:
        pmodel: a scikit-learn single-output regression estimator

    Returns:
        MultiOutputRegressor: scikit-learn multi-output regression wrapper
    """
    return MultiOutputRegressor(pmodel)
