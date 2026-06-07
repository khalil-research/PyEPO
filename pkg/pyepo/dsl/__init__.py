#!/usr/bin/env python
"""
PyEPO symbolic DSL: define an LP / MIP / QP once and compile it to any backend.

Public API::

    from pyepo import EPO, dsl
    x = dsl.Variable(5, vtype=EPO.BINARY)
    c = dsl.Parameter(5)
    prob = dsl.Problem(dsl.Maximize(c @ x), [W @ x <= cap])

``Problem`` finalizes the symbolic IR (sparse coefficient matrices) and
compiles it to a solver backend.
"""

from __future__ import annotations

from pyepo.dsl.expression import Parameter, Variable
from pyepo.dsl.objective import Maximize, Minimize
from pyepo.dsl.problem import Problem

__all__ = [
    "Maximize",
    "Minimize",
    "Parameter",
    "Problem",
    "Variable",
    "sum",
]


def sum(expr, axis=None):
    """
    Module-level reduction: dispatches to the expression's ``.sum(axis)``.

    ``dsl.sum(c * x)`` returns the scalar ``ParametricBilinear``; ``dsl.sum`` of
    an ``Affine`` / ``Variable`` returns the reduced linear expression.
    """
    return expr.sum(axis)
