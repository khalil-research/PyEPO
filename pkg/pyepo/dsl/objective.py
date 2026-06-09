#!/usr/bin/env python
"""
Objective nodes for the PyEPO DSL.

``Minimize`` / ``Maximize`` wrap a scalar ``ParametricObjective`` (the predicted
cost paired with a Variable, optionally plus a parameter-free quadratic term)
and record the ``EPO`` model sense for the compiled backend.
"""

from __future__ import annotations

from pyepo import EPO


class Objective:
    """
    Base objective: a wrapped expression plus its sense.

    Attributes:
        expr (ParametricObjective): the objective expression
        modelSense (EPO): ``EPO.MINIMIZE`` or ``EPO.MAXIMIZE``
    """

    modelSense = None

    def __init__(self, expr):
        from pyepo.dsl.expression import ParametricObjective, ParametricVector

        # an elementwise c * x is a vector, not a scalar objective
        if isinstance(expr, ParametricVector):
            raise TypeError(
                "c * x is an elementwise vector, not a scalar objective; "
                "write (c * x).sum() or c @ x."
            )
        # the objective must carry the predicted cost (a ParametricObjective)
        if not isinstance(expr, ParametricObjective):
            raise TypeError(
                "The objective must be a predicted cost term like c @ x, "
                "optionally plus a known quadratic term."
            )
        self.expr = expr

    @property
    def cost_param(self):
        return self.expr.cost_param

    @property
    def cost_var(self):
        return self.expr.cost_var


class Minimize(Objective):
    """Minimization objective."""

    modelSense = EPO.MINIMIZE


class Maximize(Objective):
    """Maximization objective."""

    modelSense = EPO.MAXIMIZE
