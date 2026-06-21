#!/usr/bin/env python
"""
OR-Tools (pywraplp) compiler for the PyEPO DSL.

``compiledOrtProblem`` mixes the generic ``compiledBase`` with ``optOrtModel``
to turn a finalized DSL ``Problem`` into a pywraplp ``Solver`` running an open
solver (``solver=``, default SCIP). It builds the model and provides the OR-Tools
read / write hooks; the objective handling lives in ``compiledBase``. pywraplp is
LP / MIP only — quadratic objectives or constraints raise ``NotImplementedError``.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

try:
    from ortools.linear_solver import pywraplp
except ImportError:
    pass

from pyepo import EPO
from pyepo.dsl.compiled import compiledBase
from pyepo.model.opt import optModel
from pyepo.model.ort.ortmodel import optOrtModel


def compileProblem(problem, solver="scip", **params) -> compiledOrtProblem:
    """Instantiate the OR-Tools-compiled problem; ``params`` are solver options."""
    return compiledOrtProblem(problem, params=params, solver=solver)


class compiledOrtProblem(compiledBase, optOrtModel):
    """
    OR-Tools (pywraplp) compiled DSL problem, solved by ``solver``.
    """

    def __init__(self, problem, params=None, solver="scip"):
        # the source DSL Problem, solver options, and pywraplp backend name
        self.problem = deepcopy(problem)
        self.params = dict(params) if params else {}
        self.solver = solver
        self._extra_constrs = []  # (coef, rhs) cuts replayed on copy
        optModel.__init__(self)  # builds the model via _getModel
        self._model.SuppressOutput()
        self._set_obj_sense()
        self._vars_list = list(self.x.values())
        self._apply_params()

    def _getModel(self) -> tuple:
        # build the pywraplp model from the finalized IR
        prob = self.problem
        if prob.obj_Q is not None:
            raise NotImplementedError("OR-Tools (pywraplp) does not support quadratic objectives.")
        self.modelSense = prob.modelSense
        m = pywraplp.Solver.CreateSolver(self.solver.upper())
        if m is None:
            raise RuntimeError(f"OR-Tools solver {self.solver!r} is not available.")
        x = self._build_flat_vars(m)
        self._emit_constraints(m, x)
        return m, x

    def _apply_params(self):
        # apply solver options; the canonical `timelimit` (seconds) -> milliseconds
        for key, value in self.params.items():
            if key.lower() == "timelimit":
                self._model.SetTimeLimit(int(value * 1000))
            else:
                raise ValueError(
                    f"OR-Tools backend supports only the 'timelimit' param, got {key!r}."
                )

    def _write_obj(self, coef):
        # set the full-length objective coefficient
        obj = self._model.Objective()
        for j in range(self.problem.num_vars):
            obj.SetCoefficient(self.x[j], float(coef[j]))

    def _read_sol(self):
        # solve and read the full solution + objective value
        status = self._model.Solve()
        # FEASIBLE keeps the time-limited incumbent usable
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            raise RuntimeError(f"OR-Tools found no solution (status {status}).")
        sol = np.fromiter(
            (self.x[j].solution_value() for j in range(self.problem.num_vars)), dtype=float
        )
        return sol, self._model.Objective().Value()

    def _add_cut(self, coef, rhs):
        # pywraplp has no clone; copy() rebuilds + replays cuts, then add the new one
        new_model = self.copy()
        new_model._extra_constrs.append((np.asarray(coef, dtype=float), float(rhs)))
        ct = new_model._model.Constraint(-new_model._model.infinity(), float(rhs))
        for j in range(self.problem.num_vars):
            ct.SetCoefficient(new_model.x[j], float(coef[j]))
        return new_model

    def _build_flat_vars(self, m):
        # one Var per entry, keyed by flat index, with per-entry domain and bounds
        prob = self.problem
        inf = m.infinity()
        x = {}
        for j in range(prob.num_vars):
            t = prob.var_type[j]
            if t == EPO.BINARY:
                x[j] = m.BoolVar(f"x_{j}")
                continue
            lb = -inf if np.isneginf(prob.var_lb[j]) else float(prob.var_lb[j])
            ub = inf if np.isposinf(prob.var_ub[j]) else float(prob.var_ub[j])
            x[j] = m.IntVar(lb, ub, f"x_{j}") if t == EPO.INTEGER else m.NumVar(lb, ub, f"x_{j}")
        return x

    def _emit_constraints(self, m, x):
        # linear constraints from the finalized IR (one relation per row)
        for Q, A, sense, b in self.problem.constrs:
            if Q is not None:
                raise NotImplementedError(
                    "OR-Tools (pywraplp) does not support quadratic constraints."
                )
            A = A.tocsr()
            b = np.asarray(b, dtype=float).reshape(-1)
            for r in range(A.shape[0]):
                row = A.getrow(r)
                lhs = sum(float(v) * x[j] for j, v in zip(row.indices, row.data))
                if sense == "<=":
                    m.Add(lhs <= b[r])
                elif sense == ">=":
                    m.Add(lhs >= b[r])
                else:
                    m.Add(lhs == b[r])
