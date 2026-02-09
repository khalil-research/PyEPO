#!/usr/bin/env python
# coding: utf-8

"""
Utility function
"""

import inspect

def getArgs(model):
    """
    A global function to get args of model

    Args:
        model (optModel): optimization model

    Returns:
        dict: model args
    """
    params = inspect.signature(model.__init__).parameters
    return {name: getattr(model, name) for name in params if hasattr(model, name)}
