#!/usr/bin/env python
"""
COPT compiler for the PyEPO DSL.

``compiledCoptProblem`` mixes the generic ``compiledBase`` with ``optCoptModel``
to turn a finalized DSL ``Problem`` into a Cardinal Optimizer model. It builds
the model and provides the COPT read / write hooks; the objective handling lives
in ``compiledBase``.
"""

from __future__ import annotations

import numpy as np

try:
    from coptpy import COPT
except ImportError:
    COPT = None

from pyepo import EPO
from pyepo.dsl.compiled import compiledBase
from pyepo.model.copt.coptmodel import _get_envr, _read_solution, optCoptModel


def compileProblem(problem, **params) -> compiledCoptProblem:
    """Instantiate the COPT-compiled problem; ``params`` are COPT parameters."""
    return compiledCoptProblem(problem, params=params)


class compiledCoptProblem(compiledBase, optCoptModel):
    """
    COPT-backed compiled DSL problem.
    """

    def _getModel(self) -> tuple:
        # build the COPT model from the finalized IR
        prob = self.problem
        m = _get_envr().createModel("dsl")
        # objective sense (EPO -> COPT)
        sense = COPT.MAXIMIZE if prob.modelSense == EPO.MAXIMIZE else COPT.MINIMIZE
        m.setObjSense(sense)
        x = self._build_flat_vars(m)
        self._emit_constraints(m, x)
        # parameter-free quadratic objective term
        if prob.obj_Q is not None:
            m.setObjective(x @ prob.obj_Q @ x)
        return m, x

    def _apply_params(self):
        # apply solver params; the canonical `timelimit` (seconds) maps to TimeLimit
        for key, value in self.params.items():
            self._model.setParam("TimeLimit" if key == "timelimit" else key, value)

    def _write_obj(self, coef):
        # set the full-length objective coefficient on the MVar
        self.x.Obj = coef

    def _read_sol(self):
        # optimize and read the full solution + objective value
        self._model.solve()
        return _read_solution(
            self._model,
            lambda: np.asarray(self.x.x.tolist(), dtype=float),
        )

    def _add_cut(self, coef, rhs):
        # add coef @ x <= rhs to a fresh copy
        new_model = self.copy()
        new_model._model.addConstr(coef @ new_model.x <= float(rhs))
        return new_model

    def _build_flat_vars(self, m):
        # one MVar with per-entry bounds and type
        prob = self.problem
        lb = np.where(np.isneginf(prob.var_lb), -COPT.INFINITY, prob.var_lb)
        ub = np.where(np.isposinf(prob.var_ub), COPT.INFINITY, prob.var_ub)
        # EPO type -> COPT vtype
        copt_vtype = {
            EPO.BINARY: COPT.BINARY,
            EPO.INTEGER: COPT.INTEGER,
            EPO.CONTINUOUS: COPT.CONTINUOUS,
        }
        vtype = [copt_vtype[t] for t in prob.var_type]
        return m.addMVar(
            prob.num_vars, lb=lb, ub=ub, vtype=vtype, nameprefix=prob.cost_var_name or "x"
        )

    def _emit_constraints(self, m, x):
        # linear (Q is None) via addConstr, quadratic via addQConstr
        for i, (Q, A, sense, b) in enumerate(self.problem.constrs):
            if Q is None:
                expr = A @ x
                rhs = np.asarray(b, dtype=float)
                add = m.addConstr
            else:
                a = np.asarray(A.todense(), dtype=float).reshape(-1)
                expr = x @ Q @ x + (a @ x if a.any() else 0.0)
                rhs = float(np.asarray(b, dtype=float).reshape(-1)[0])
                add = m.addQConstr
            # name each constraint group
            name = f"c{i}"
            if sense == "<=":
                add(expr <= rhs, name=name)
            elif sense == ">=":
                add(expr >= rhs, name=name)
            else:
                add(expr == rhs, name=name)
