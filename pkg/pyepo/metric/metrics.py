#!/usr/bin/env python
"""
Metrics for SKlearn model
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable

import numpy as np

from pyepo.metric.regret import _checkLinearObj, calRegret
from pyepo.utils import _EPS, getArgs

if TYPE_CHECKING:
    from pyepo.model.opt import optModel


def SPOError(
    pred_cost: np.ndarray,
    true_cost: np.ndarray,
    optmodel: optModel | type,
    args: dict | None = None,
) -> float:
    """
    Normalized true regret of predicted costs over a dataset.

    Solves each instance at the predicted and the true cost and returns
    :math:`\\sum_i l_i / \\sum_i |z^*(\\mathbf{c}_i)|`; instances with
    near-zero true optima inflate the ratio.

    Args:
        pred_cost: predicted costs of shape (num_data, num_cost)
        true_cost: true costs of shape (num_data, num_cost)
        optmodel: a PyEPO optimization model
        args: optModel args for the deprecated ``(model_type, args)`` form

    Returns:
        float: normalized regret
    """
    # deprecated form: rebuild the model from its class and args
    if isinstance(optmodel, type):
        warnings.warn(
            "Passing (model_type, args) to SPOError is deprecated; pass an optModel instance.",
            DeprecationWarning,
            stacklevel=2,
        )
        optmodel = optmodel(**(args or {}))
    _checkLinearObj(optmodel)
    pred_cost = np.array(pred_cost)
    true_cost = np.array(true_cost)
    assert pred_cost.shape == true_cost.shape, "Shape of true and predicted value does not match."
    # init sum
    regret_sum = 0.0
    optobj_sum = 0.0
    for c, cp in zip(true_cost, pred_cost):
        # opt obj for true cost
        optmodel._setFullObj(optmodel._fullCost(c))
        _, optobj = optmodel.solve()
        # per-instance regret
        regret_sum += calRegret(optmodel, cp, c, optobj)
        optobj_sum += np.abs(optobj)
    # normalized regret
    return regret_sum / (optobj_sum + _EPS)


def _SPOErrorScore(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_type: type,
    args: dict,
) -> float:
    """
    A function to calculate normalized true regret with the (y_true, y_pred)
    argument order of sklearn / autosklearn scorers

    Args:
        y_true: true costs
        y_pred: predicted costs
        model_type: optModel class type
        args: optModel args

    Returns:
        float: regret loss
    """
    # rebuild per call; class + args stay picklable for scorer kwargs
    return SPOError(y_pred, y_true, model_type(**args))


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
    SPO_scorer = make_scorer(
        _SPOErrorScore, greater_is_better=False, model_type=model_type, args=args
    )
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
        score_func=_SPOErrorScore,
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
