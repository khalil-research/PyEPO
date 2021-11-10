#!/usr/bin/env python
# coding: utf-8
"""
Two-stage model with Scikit-learn predictor
"""

import numpy as np
from autosklearn.regression import AutoSklearnRegressor


def autoSklearnPred():
    """
    Two-stage prediction and optimization with auto-sklearn.
    """
    predictor = AutoSklearnRegressor(time_left_for_this_task=600,
                                     per_run_time_limit=30)
    return predictor
