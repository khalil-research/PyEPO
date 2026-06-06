#!/usr/bin/env python
"""
PyEPO symbolic DSL: define an LP / MIP / QP once and compile it to any backend.

Public API::

    from pyepo import dsl
    x = dsl.Variable(5, binary=True)
    c = dsl.Parameter(5)
    prob = dsl.Problem(dsl.Maximize(c @ x), [W @ x <= cap])

The Phase 1 core builds and finalizes the symbolic IR (sparse coefficient
matrices, cost layout); backend compilation is Phase 2.
"""

from __future__ import annotations

from pyepo.dsl.constraint import Constraint
from pyepo.dsl.expression import (
    Affine,
    Parameter,
    ParametricBilinear,
    Quadratic,
    Variable,
    quadForm,
)
from pyepo.dsl.objective import Maximize, Minimize, Objective
from pyepo.dsl.problem import Problem

__all__ = [
    "Affine",
    "Constraint",
    "Maximize",
    "Minimize",
    "Objective",
    "Parameter",
    "ParametricBilinear",
    "Problem",
    "Quadratic",
    "Variable",
    "quadForm",
    "sum",
]


def sum(expr, axis=None):
    """
    Module-level reduction: dispatches to the expression's ``.sum(axis)``.

    ``dsl.sum(c * x)`` returns the scalar ``ParametricBilinear``; ``dsl.sum`` of
    an ``Affine`` / ``Variable`` returns the reduced linear expression.
    """
    return expr.sum(axis)
