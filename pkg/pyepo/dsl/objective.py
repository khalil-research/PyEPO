#!/usr/bin/env python
"""
Objective nodes for the PyEPO DSL.

``Minimize`` / ``Maximize`` wrap a scalar ``ParametricBilinear`` (the predicted
cost paired with a Variable, optionally plus a parameter-free quadratic term)
and record the ``EPO`` model sense for the compiled backend.
"""

from __future__ import annotations

from pyepo import EPO


class Objective:
    """
    Base objective: a wrapped expression plus its sense.

    Attributes:
        expr (ParametricBilinear): the objective expression
        modelSense (EPO): ``EPO.MINIMIZE`` or ``EPO.MAXIMIZE``
    """

    modelSense = None

    def __init__(self, expr):
        from pyepo.dsl.expression import ParametricBilinear
        # the objective must carry the predicted cost (a ParametricBilinear)
        if not isinstance(expr, ParametricBilinear):
            raise TypeError("Objective must be a ParametricBilinear (e.g. c @ x), optionally + a "
                            "parameter-free quadratic term.")
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
