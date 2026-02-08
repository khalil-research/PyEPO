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

    Return:
        dict: model args
    """
    for mem in inspect.getmembers(model):
        if mem[0] == "__dict__":
            attrs = mem[1]
            args = {}
            for name in attrs:
                if name in inspect.signature(model.__init__).parameters:
                    args[name] = attrs[name]
            return args
