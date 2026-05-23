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
    Two-stage prediction and optimization with scikit-learn.

    Args:
        pmodel (Regressor): scikit-learn regression model

    Returns:
        MultiOutputRegressor: scikit-learn multi-output regression model
    """
    return MultiOutputRegressor(pmodel)
