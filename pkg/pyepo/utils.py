#!/usr/bin/env python

"""
Utility function
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyepo.model.opt import optModel


def getArgs(model: optModel) -> dict:
    """
    A global function to get args of model

    Args:
        model (optModel): optimization model

    Returns:
        dict: model args
    """
    params = inspect.signature(model.__init__).parameters
    return {name: getattr(model, name) for name in params if hasattr(model, name)}
