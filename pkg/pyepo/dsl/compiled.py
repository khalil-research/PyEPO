#!/usr/bin/env python
"""
Generic compiled-problem base for the PyEPO DSL.

``compiledBase`` is mixed with a backend base (``optXxxModel``) to form a
concrete compiled problem, e.g. ``compiledGrbProblem(compiledBase,
optGrbModel)`` — the same composition as ``knapsackModel(knapsackBase,
optGrbModel)``. The DSL core targets problems whose predicted cost is 1:1 with
the decision variables (``num_vars == num_cost``), so ``setObj`` / ``solve`` /
``addConstr`` / ``copy`` are inherited from the backend base unchanged;
``compiledBase`` only carries ``num_cost`` and ``relax``. Problems with
auxiliary variables or cost broadcast (e.g. TSP/VRP) are hand-written as a
backend subclass that overrides ``setObj`` / ``solve``.
"""

from __future__ import annotations

from pyepo.model.opt import optModel
from pyepo.utils import getArgs


class compiledBase(optModel):
    """
    Backend-agnostic compiled DSL problem (1:1 cost). Mixed with an
    ``optXxxModel``; the concrete subclass supplies ``_getModel`` and
    ``_apply_params``.
    """

    def __init__(self, problem, params=None):
        # the source DSL Problem and backend solver parameters
        self.problem = problem
        self.params = dict(params) if params else {}
        # the predicted cost must cover every decision variable
        if problem.num_vars != problem.num_cost:
            raise ValueError(
                f"The predicted cost must cover every decision variable, but this "
                f"problem has {problem.num_vars} variable(s) and the cost covers "
                f"{problem.num_cost}. Variables that appear only in constraints are "
                f"not supported."
            )
        super().__init__()
        self._apply_params()

    def _apply_params(self):
        # backend hook: push self.params to the solver
        pass

    @property
    def num_cost(self) -> int:
        # predicted cost dimension = the unique Parameter's size
        return self.problem.num_cost

    def relax(self):
        # recompile the relaxed problem
        kwargs = getArgs(self)
        kwargs["problem"] = self.problem.relax()
        return type(self)(**kwargs)
