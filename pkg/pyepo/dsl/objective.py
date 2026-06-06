#!/usr/bin/env python
"""
Objective nodes for the PyEPO DSL.

``Minimize`` / ``Maximize`` wrap a scalar ``ParametricBilinear`` (the predicted
cost paired with a Variable, optionally plus a parameter-free quadratic /
constant offset) and record both the string ``sense`` and the matching
``EPO`` sense for the compiled backend.
"""

from __future__ import annotations

from pyepo import EPO


class Objective:
    """
    Base objective: a wrapped expression plus its sense.

    Attributes:
        expr (ParametricBilinear): the objective expression
        sense (str): ``"minimize"`` or ``"maximize"``
        epo_sense (EPO): matching ``EPO.MINIMIZE`` / ``EPO.MAXIMIZE``
    """

    sense = None
    epo_sense = None

    def __init__(self, expr):
        from pyepo.dsl.expression import ParametricBilinear
        # promote a bare Variable/Affine objective would be parameter-free — reject early
        if not isinstance(expr, ParametricBilinear):
            raise TypeError("Objective must be a ParametricBilinear (e.g. c @ x), optionally + a "
                            "parameter-free quadratic / constant offset.")
        self.expr = expr

    @property
    def cost_param(self):
        return self.expr.cost_param

    @property
    def cost_var(self):
        return self.expr.cost_var


class Minimize(Objective):
    """Minimization objective."""
    sense = "minimize"
    epo_sense = EPO.MINIMIZE


class Maximize(Objective):
    """Maximization objective."""
    sense = "maximize"
    epo_sense = EPO.MAXIMIZE
