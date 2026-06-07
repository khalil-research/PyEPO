#!/usr/bin/env python
"""
Generic compiled-problem base for the PyEPO DSL.

``compiledBase`` is mixed with a backend base (``optXxxModel``) to form a
concrete compiled problem, e.g. ``compiledGrbProblem(compiledBase,
optGrbModel)``. It carries the backend-agnostic objective handling — binding
the predicted cost onto its positions, solving, and projecting the solution
back to cost space — while the concrete subclass builds the solver model and
provides the read / write hooks.
"""

from __future__ import annotations

import numpy as np

from pyepo.model.opt import optModel
from pyepo.utils import costToNumpy, getArgs


class compiledBase(optModel):
    """
    Backend-agnostic compiled DSL problem. Mixed with an ``optXxxModel``.
    """

    def __init__(self, problem, params=None):
        # the source DSL Problem and backend solver parameters
        self.problem = problem
        self.params = dict(params) if params else {}
        super().__init__()
        self._apply_params()

    @property
    def num_cost(self) -> int:
        # predicted cost dimension
        return self.problem.num_cost

    def setObj(self, c):
        # objective coef = known fixed base, plus the predicted cost at its positions
        prob = self.problem
        coef = prob.fixed_c.copy()
        coef[prob.c_index] += costToNumpy(c)
        self._write_obj(coef)

    def solve(self):
        # solve, project onto the predicted positions, drop the fixed-cost part
        sol, obj = self._read_sol()
        sol = np.asarray(sol)
        return sol[self.problem.c_index], obj - float(self.problem.fixed_c @ sol)

    def addConstr(self, coefs, rhs):
        # a cut over the predicted positions, scattered to the full variable vector
        prob = self.problem
        full = np.zeros(prob.num_vars)
        full[prob.c_index] = np.asarray(coefs, dtype=float).reshape(-1)
        return self._add_cut(full, rhs)

    def relax(self):
        # recompile the relaxed problem, preserving backend kwargs
        kwargs = getArgs(self)
        kwargs["problem"] = self.problem.relax()
        return type(self)(**kwargs)

    def _apply_params(self):
        # push self.params to the solver
        pass

    def _write_obj(self, coef):
        # set the full-length objective coefficient vector
        raise NotImplementedError

    def _read_sol(self):
        # optimize and return (full solution, objective value)
        raise NotImplementedError

    def _add_cut(self, coef, rhs):
        # add ``coef @ x <= rhs`` to a fresh copy and return it
        raise NotImplementedError
