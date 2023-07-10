#!/usr/bin/env python
# coding: utf-8
"""
Two-stage model with Scikit-learn predictor
"""

import numpy as np
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import mean_squared_error
from autosklearn.pipeline.components import data_preprocessing
from ConfigSpace.configuration_space import ConfigurationSpace
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, INPUT

from pyepo.metric import makeAutoSkScorer

class NoPreprocessing(AutoSklearnPreprocessingAlgorithm):
    """
    This is class of NoPreprocessing component for auto-sklearn
    """
    def __init__(self, **kwargs):
        """
        This preprocessors does not change the data
        """
        # Some internal checks makes sure parameters are set
        for key, val in kwargs.items():
            setattr(self, key, val)

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "NoPreprocessing",
            "name": "NoPreprocessing",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (INPUT,)
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        return ConfigurationSpace()  # Return an empty configuration as there is None


def autoSklearnPred(optmodel, seed, timelimit, metric="mse"):
    """
    Two-stage prediction and optimization with auto-sklearn.

    Args:
        optmodel (optModel): optimization model

    Returns:
        AutoSklearnRegressor: Auto-SKlearn multi-output regression model
    """
    # add NoPreprocessing component to auto-sklearn.
    data_preprocessing.add_preprocessor(NoPreprocessing)
    # get metrics
    pyepo_scorer = makeAutoSkScorer(optmodel)
    #scorer = makeTestMSEScorer(optmodel)
    # build regressor
    if metric == "regret":
        regressor = AutoSklearnRegressor(time_left_for_this_task=timelimit,
                                         per_run_time_limit=1200,
                                         memory_limit=None,
                                         seed=seed,
                                         metric=pyepo_scorer,
                                         #scoring_functions=[pyepo_scorer, mean_squared_error],
                                         include={"data_preprocessor": ["NoPreprocessing"],
                                                  "feature_preprocessor": ["no_preprocessing"]})
                                                  #"regressor": ["adaboost", "ard_regression", "extra_trees",
                                                    #            "gaussian_process", "k_nearest_neighbors",
                                                    #            "mlp", "random_forest"]})
    elif metric == "mse":
        regressor = AutoSklearnRegressor(time_left_for_this_task=timelimit,
                                         per_run_time_limit=150,
                                         memory_limit=None,
                                         seed=seed,
                                         metric=mean_squared_error,
                                         include={"data_preprocessor": ["NoPreprocessing"],
                                                  "feature_preprocessor": ["no_preprocessing"]})
    else:
        raise ValueError("Invalid metric {}. Metric should be 'regret' or 'mse'.".format(metric))
    return regressor
