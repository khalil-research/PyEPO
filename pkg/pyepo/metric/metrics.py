#!/usr/bin/env python
"""
Metrics for SKlearn model
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from pyepo.metric._common import normalize_regret, validate_numpy_cost_batches
from pyepo.metric.regret import _checkLinearObj, calRegret

if TYPE_CHECKING:
    from pyepo.model.opt import ModelSpec, optModel


def SPOError(
    pred_cost: np.ndarray,
    true_cost: np.ndarray,
    optmodel: optModel,
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

    Returns:
        float: normalized regret
    """
    pred_cost, true_cost = validate_numpy_cost_batches(pred_cost, true_cost, optmodel.num_cost)
    _checkLinearObj(optmodel)
    # init sum
    regret_sum = 0.0
    optobj_sum = 0.0
    for c, cp in zip(true_cost, pred_cost):
        # opt obj for true cost
        optmodel._setFullObj(optmodel._fullCost(c))
        _, optobj = optmodel.solve()
        # per-instance regret
        regret_sum += calRegret(optmodel, cp, c, float(optobj))
        optobj_sum += np.abs(optobj)
    # normalized regret
    return normalize_regret(regret_sum, optobj_sum)


def _SPOErrorScore(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_spec: ModelSpec,
) -> float:
    """
    A function to calculate normalized true regret with the (y_true, y_pred)
    argument order of sklearn / autosklearn scorers

    Args:
        y_true: true costs
        y_pred: predicted costs
        model_spec: serializable optimization-model rebuild recipe

    Returns:
        float: regret loss
    """
    # rebuild per call; the spec stays picklable for scorer kwargs
    return SPOError(y_pred, y_true, model_spec.build())


def makeSkScorer(optmodel: optModel) -> Callable:
    """
    A function to create sklearn scorer

    Args:
        optmodel: optimization model

    Returns:
        scorer: callable object that returns a scalar score; less is better.
    """
    from sklearn.metrics import make_scorer

    # build score
    SPO_scorer = make_scorer(_SPOErrorScore, greater_is_better=False, model_spec=optmodel.to_spec())
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

    # build score
    SPO_scorer = make_scorer(
        name="SPO_error",
        score_func=_SPOErrorScore,
        greater_is_better=False,
        needs_proba=False,
        needs_threshold=False,
        model_spec=optmodel.to_spec(),
    )
    return SPO_scorer
