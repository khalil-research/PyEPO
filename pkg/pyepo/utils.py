#!/usr/bin/env python

"""
Package-wide utility functions
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from pyepo.model.opt import optModel


# defensive denominator regularizer (gradient / regret normalization)
_EPS: float = 1e-8


def objective_offset(model: optModel) -> float:
    """
    Bare objective constant of a compiled DSL problem.

    Args:
        model: optimization model

    Returns:
        float: objective constant, 0.0 for models without one
    """
    problem = getattr(model, "problem", None)
    return float(problem.obj_offset) if problem is not None else 0.0


def require_linear_objective(optmodel: optModel) -> None:
    """
    Reject compiled DSL models carrying a quadratic objective term.

    Args:
        optmodel: optimization model
    """
    problem = getattr(optmodel, "problem", None)
    if problem is not None and getattr(problem, "obj_Q", None) is not None:
        raise ValueError(
            "A quadratic objective term is solve-only; "
            "training methods and regret metrics require a linear objective."
        )


def costToNumpy(
    c: np.ndarray | torch.Tensor | list,
    dtype: Any = np.float32,
) -> np.ndarray:
    """
    Normalize a cost vector to a numpy array, detaching torch tensors as needed.

    Args:
        c: cost vector
        dtype: target dtype when ``c`` is not already a tensor; torch
            tensors are converted via ``.detach().cpu().numpy()`` and keep their
            existing dtype.

    Returns:
        np.ndarray: numpy cost vector
    """
    if isinstance(c, torch.Tensor):
        return c.detach().cpu().numpy()
    return np.asarray(c, dtype=dtype)
