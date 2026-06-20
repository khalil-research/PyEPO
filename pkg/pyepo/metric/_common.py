"""Shared helpers for metric configuration and evaluation."""

from __future__ import annotations

import math
from contextlib import contextmanager
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import torch

from pyepo.utils import _EPS

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import nn


@contextmanager
def torch_evaluation(model: nn.Module | None) -> Iterator[torch.device]:
    """Yield the model device while temporarily enabling evaluation mode."""
    if model is None:
        yield torch.device("cpu")
        return

    was_training = model.training
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    model.eval()
    try:
        yield device
    finally:
        model.train(was_training)


def validate_tolerance(tolerance: float) -> None:
    """Validate a finite, strictly positive numerical tolerance."""
    if not isinstance(tolerance, Real) or isinstance(tolerance, bool):
        raise ValueError("tolerance must be a finite positive number.")
    number = float(tolerance)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("tolerance must be a finite positive number.")


def validate_retry_count(max_iter: int) -> None:
    """Validate the recursive retry budget while preserving exhaustion semantics."""
    if not isinstance(max_iter, int) or isinstance(max_iter, bool):
        raise ValueError("max_iter must be a positive integer.")
    if max_iter <= 0:
        raise RuntimeError("Max iterations reached in calUnambRegret.")


def is_real_numeric_array(value: np.ndarray) -> bool:
    """Return whether an array has a non-complex numerical dtype."""
    return np.issubdtype(value.dtype, np.number) and not np.issubdtype(
        value.dtype, np.complexfloating
    )


def normalize_regret(regret_sum: float, absolute_optimum_sum: float) -> float:
    """Normalize aggregate regret by absolute true optimum magnitude."""
    return float(regret_sum) / (float(absolute_optimum_sum) + _EPS)


def validate_cost_vectors(
    pred_cost,
    true_cost,
    true_obj,
    num_cost: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return valid single-instance costs and objective value."""
    pred = np.asarray(pred_cost)
    true = np.asarray(true_cost)
    if pred.ndim != 1 or true.ndim != 1:
        raise ValueError("Predicted and true costs must be one-dimensional vectors.")
    if pred.shape != true.shape:
        raise ValueError("Shape of true and predicted cost does not match.")
    if pred.shape[0] != num_cost:
        raise ValueError(f"Cost vector length must match optmodel.num_cost ({num_cost}).")
    if not is_real_numeric_array(pred) or not is_real_numeric_array(true):
        raise ValueError("Predicted and true costs must be numerical vectors.")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("Predicted and true costs must contain only finite values.")
    if not isinstance(true_obj, Real) or isinstance(true_obj, bool):
        raise ValueError("true_obj must be a finite number.")
    objective = float(true_obj)
    if not math.isfinite(objective):
        raise ValueError("true_obj must be a finite number.")
    return pred, true, objective
