#!/usr/bin/env python
"""
Metrics for SKlearn model
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from pyepo.metric._common import is_real_numeric_array
from pyepo.metric.regret import _checkLinearObj, calRegret
from pyepo.utils import _EPS

if TYPE_CHECKING:
    from pyepo.model.opt import ModelSpec, optModel


def _validate_cost_batches(
    pred_cost: np.ndarray,
    true_cost: np.ndarray,
    num_cost: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, matching two-dimensional cost batches."""
    try:
        pred = np.asarray(pred_cost)
        true = np.asarray(true_cost)
    except (TypeError, ValueError) as exc:
        raise ValueError("Predicted and true costs must be numerical arrays.") from exc

    if pred.shape != true.shape:
        raise ValueError("Shape of true and predicted value does not match.")
    if pred.ndim != 2:
        raise ValueError("Predicted and true costs must be two-dimensional batches.")
    if pred.shape[0] == 0:
        raise ValueError("Predicted and true cost batches must not be empty.")
    if pred.shape[1] != num_cost:
        raise ValueError(f"Cost batch width must match optmodel.num_cost ({num_cost}).")
    if not is_real_numeric_array(pred) or not is_real_numeric_array(true):
        raise ValueError("Predicted and true costs must be numerical arrays.")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("Predicted and true costs must contain only finite values.")
    return pred, true


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
    pred_cost, true_cost = _validate_cost_batches(pred_cost, true_cost, optmodel.num_cost)
    _checkLinearObj(optmodel)
    # init sum
    regret_sum = 0.0
    optobj_sum = 0.0
    for c, cp in zip(true_cost, pred_cost):
        # opt obj for true cost
        optmodel._setFullObj(optmodel._fullCost(c))
        _, optobj = optmodel.solve()
        # per-instance regret
        regret_sum += calRegret(optmodel, cp, c, optobj)  # type: ignore[arg-type]
        optobj_sum += np.abs(optobj)
    # normalized regret
    return regret_sum / (optobj_sum + _EPS)


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
