#!/usr/bin/env python
# coding: utf-8
"""
Two-stage model with Scikit-learn predictor
"""

import numpy as np
from sklearn.multioutput import MultiOutputRegressor


def sklearnPred(pmodel):
    """
    Two-stage prediction and optimization with scikit-learn.

    Args:
        pmodel (Regressor): scikit-learn regression model

    Returns:
        MultiOutputRegressor: scikit-learn multi-output regression model
    """
    return MultiOutputRegressor(pmodel)
