#!/usr/bin/env python
"""
PyEPO-scope validation for a DSL ``Problem``.

The restricted ``Parameter`` overloads already reject ill-formed objectives at
construction, so validation is a small set of structural assertions: exactly
one predicted ``Parameter``, only in the objective, and an objective that is a
``ParametricBilinear``.
"""

from __future__ import annotations


def validate(problem) -> None:
    """
    Run the PyEPO-scope checks; raise ``ValueError`` / ``TypeError`` on failure.
    """
    from pyepo.dsl.expression import ParametricBilinear
    from pyepo.dsl.objective import Objective

    # §7.3 objective is a ParametricBilinear (Objective.__init__ enforces; re-assert)
    if not isinstance(problem.objective, Objective):
        raise TypeError("Problem objective must be a Minimize / Maximize.")
    if not isinstance(problem.objective.expr, ParametricBilinear):
        raise TypeError("Objective must be a ParametricBilinear (e.g. c @ x).")

    # §7.1 exactly one Parameter (the objective's cost_param)
    if problem.cost_param is None:
        raise ValueError("Problem has no Parameter (the predicted cost vector).")

    # §7.2 Parameter only in the objective
    for con in problem.constraints:
        if con.has_parameter():
            raise ValueError("A Parameter appears in a constraint; it may only appear in the objective.")

    # cost layout: 1:1 needs equal sizes; broadcast needs a valid gather
    obj = problem.objective.expr
    if obj.cost_gather is None:
        if obj.cost_param.size != obj.cost_var.size:
            raise ValueError(
                f"cost_param size {obj.cost_param.size} != cost_var size "
                f"{obj.cost_var.size} (use a fixed gather to broadcast)."
            )
    else:
        if len(obj.cost_gather) != obj.cost_var.size:
            raise ValueError(
                f"cost_gather length {len(obj.cost_gather)} != cost_var size {obj.cost_var.size}."
            )
        if obj.cost_gather.min() < 0 or obj.cost_gather.max() >= obj.cost_param.size:
            raise ValueError("cost_gather index out of range of the Parameter.")
