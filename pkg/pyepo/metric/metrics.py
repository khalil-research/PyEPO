#!/usr/bin/env python
"""
Metrics for SKlearn model
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from pyepo import EPO
from pyepo.utils import _EPS, getArgs

if TYPE_CHECKING:
    from pyepo.model.opt import optModel


def SPOError(
    pred_cost: np.ndarray,
    true_cost: np.ndarray,
    model_type: type,
    args: dict,
) -> float:
    """
    A function to calculate normalized true regret

    Args:
        pred_cost: predicted costs
        true_cost: true costs
        model_type: optModel class type
        args: optModel args

    Returns:
        float: regret loss
    """
    pred_cost = np.array(pred_cost)
    true_cost = np.array(true_cost)
    assert pred_cost.shape == true_cost.shape, "Shape of true and predicted value does not match."
    # rebuild model
    optmodel = model_type(**args)
    # init sum
    regret_sum = 0
    optobj_sum = 0
    for c, cp in zip(true_cost, pred_cost):
        # opt sol for pred cost
        optmodel.setObj(optmodel._fullCost(cp))
        sol, _ = optmodel.solve()
        # full objective of the predicted decision at the true cost
        obj = np.dot(sol, optmodel._fullCost(np.asarray(c, dtype=float)))
        # opt obj for true cost
        optmodel.setObj(optmodel._fullCost(c))
        _, optobj = optmodel.solve()
        # calculate regret
        if optmodel.modelSense == EPO.MINIMIZE:
            regret = obj - optobj
        elif optmodel.modelSense == EPO.MAXIMIZE:
            regret = optobj - obj
        else:
            raise ValueError("Invalid modelSense.")
        regret_sum += regret
        optobj_sum += np.abs(optobj)
    # normalized regret
    norm_regret = regret_sum / (optobj_sum + _EPS)
    return norm_regret


def makeSkScorer(optmodel: optModel) -> Callable:
    """
    A function to create sklearn scorer

    Args:
        optmodel: optimization model

    Returns:
        scorer: callable object that returns a scalar score; less is better.
    """
    from sklearn.metrics import make_scorer

    # get class
    model_type = type(optmodel)
    # get args
    args = getArgs(optmodel)
    # build score
    SPO_scorer = make_scorer(SPOError, greater_is_better=False, model_type=model_type, args=args)
    return SPO_scorer


def makeAutoSkScorer(optmodel: optModel) -> Callable:
    """
    A function to create Auto-SKlearn scorer

    Args:
        optmodel: optimization model

    Returns:
        scorer: callable object that returns a scalar score; less is better.
    """
    from autosklearn.metrics import make_scorer  # pyright: ignore[reportMissingImports]

    # get class
    model_type = type(optmodel)
    # get args
    args = getArgs(optmodel)
    # build score
    SPO_scorer = make_scorer(
        name="SPO_error",
        score_func=SPOError,
        greater_is_better=False,
        needs_proba=False,
        needs_threshold=False,
        model_type=model_type,
        args=args,
    )
    return SPO_scorer


def testMSE(
    pred_cost: np.ndarray,
    true_cost: np.ndarray,
    model_type: type,
    args: dict,
) -> float:
    """
    A function to calculate MSE for testing

    Args:
        pred_cost: predicted costs
        true_cost: true costs
        model_type: optModel class type
        args: optModel args

    Returns:
        float: mse
    """
    pred_cost = np.array(pred_cost)
    true_cost = np.array(true_cost)
    assert pred_cost.shape == true_cost.shape, "Shape of true and predicted value does not match."
    return np.square(pred_cost - true_cost).mean()


def makeTestMSEScorer(optmodel: optModel) -> Callable:
    """
    A function to create MSE scorer for testing

    Args:
        optmodel: optimization model

    Returns:
        scorer: callable object that returns a scalar score; less is better.
    """
    from autosklearn.metrics import make_scorer  # pyright: ignore[reportMissingImports]

    # get class
    model_type = type(optmodel)
    # get args
    args = getArgs(optmodel)
    # build score
    mse_scorer = make_scorer(
        name="test_error",
        score_func=testMSE,
        greater_is_better=False,
        model_type=model_type,
        args=args,
    )
    return mse_scorer
