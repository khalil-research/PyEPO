#!/usr/bin/env python

"""
Package-wide utility functions
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from pyepo.model.opt import optModel


def getArgs(model: optModel) -> dict:
    """
    A global function to get args of model

    Args:
        model: optimization model

    Returns:
        dict: model args
    """
    params = inspect.signature(model.__init__).parameters
    return {name: getattr(model, name) for name in params if hasattr(model, name)}


def costToNumpy(c: np.ndarray | torch.Tensor | list, dtype=np.float32) -> np.ndarray:
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
