#!/usr/bin/env python
# coding: utf-8
"""
Two-stage model with Scikit-learn predictor
"""

import numpy as np
from autosklearn.regression import AutoSklearnRegressor

from spo.eval.metrics import makeAutoSkScorer

def autoSklearnPred(omodel):
    """
    Two-stage prediction and optimization with auto-sklearn.

    Args:
        omodel (optModel): optimization model

    Returns:
        AutoSklearnRegressor: Auto-SKlearn multi-output regression model
    """
    # get metrics
    spo_scorer = spo.eval.metrics.makeAutoSkScorer(omodel)
    # build regressor
    regressor = AutoSklearnRegressor(time_left_for_this_task=600,
                                     per_run_time_limit=30,
                                     seed=135,
                                     metric=spo_scorer)
    return regressor
