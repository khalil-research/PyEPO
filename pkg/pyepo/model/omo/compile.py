#!/usr/bin/env python
"""
Pyomo compiler for the PyEPO DSL.

``compiledOmoProblem`` mixes the generic ``compiledBase`` with ``optOmoModel``
to turn a finalized DSL ``Problem`` into a Pyomo ``ConcreteModel`` solved by any
Pyomo-supported ``solver``. It builds the model and provides the Pyomo read /
write hooks; the objective handling lives in ``compiledBase``. The objective
coefficient is a mutable ``Param`` so ``setObj`` updates values rather than
rebuilding the expression.
"""

from __future__ import annotations

import contextlib
from copy import deepcopy

import numpy as np

with contextlib.suppress(ImportError):
    from pyomo import environ as pe
    from pyomo import opt as po

from pyepo import EPO
from pyepo.dsl.compiled import compiledBase
from pyepo.model.omo.omomodel import _solve_model, optOmoModel
from pyepo.model.opt import optModel


def compileProblem(problem, solver="glpk", **params) -> compiledOmoProblem:
    """Instantiate the Pyomo-compiled problem; ``params`` are solver options."""
    return compiledOmoProblem(problem, params=params, solver=solver)


def _quad(Q, x):
    # sum_{i,j} Q[i,j] x[i] x[j] for a sparse symmetric Q
    Q = Q.tocoo()
    return sum(v * x[i] * x[j] for i, j, v in zip(Q.row, Q.col, Q.data))


def _add_relation(cons, lhs, sense, rhs):
    # add lhs <op> rhs to a ConstraintList
    if sense == "<=":
        cons.add(lhs <= rhs)
    elif sense == ">=":
        cons.add(lhs >= rhs)
    else:
        cons.add(lhs == rhs)


class compiledOmoProblem(compiledBase, optOmoModel):
    """
    Pyomo-backed compiled DSL problem, solved by ``solver``.
    """

    # canonical `timelimit` (seconds) -> each solver's own option name
    _TIMELIMIT_OPT = {
        "glpk": "tmlim",
        "cbc": "seconds",
        "scip": "limits/time",
        "highs": "time_limit",
        "appsi_highs": "time_limit",
        "ipopt": "max_cpu_time",
        "gurobi": "TimeLimit",
        "cplex": "timelimit",
    }

    def __init__(self, problem, params=None, solver="glpk"):
        # the source DSL Problem, solver options, and Pyomo solver name
        self.problem = deepcopy(problem)
        self.params = dict(params) if params else {}
        self.solver = solver
        optModel.__init__(self)  # builds the model via _getModel
        self._solverfac = po.SolverFactory(solver)
        self._apply_params()

    def _getModel(self) -> tuple:
        # build the Pyomo model from the finalized IR
        prob = self.problem
        self.modelSense = prob.modelSense
        m = pe.ConcreteModel()
        x = self._build_flat_vars(m)
        # objective: full coefficient as a mutable Param, plus the fixed quadratic
        m.coef = pe.Param(range(prob.num_vars), mutable=True, initialize=0.0)
        expr = sum(m.coef[j] * x[j] for j in range(prob.num_vars))
        if prob.obj_Q is not None:
            expr = expr + _quad(prob.obj_Q, x)
        sense = pe.minimize if prob.modelSense == EPO.MINIMIZE else pe.maximize
        m.obj = pe.Objective(expr=expr, sense=sense)
        self._emit_constraints(m, x)
        return m, x

    def _apply_params(self):
        # apply solver options; the canonical `timelimit` (seconds) maps per solver
        for key, value in self.params.items():
            if key == "timelimit":
                name = self._TIMELIMIT_OPT.get(self.solver)
                if name is None:
                    raise ValueError(
                        f"Pyomo solver {self.solver!r} has no known 'timelimit' option "
                        "name; pass the solver's native option instead."
                    )
                self._solverfac.options[name] = value
            else:
                self._solverfac.options[key] = value

    def _write_obj(self, coef):
        # update the mutable objective coefficient Param
        for j in range(self.problem.num_vars):
            self._model.coef[j] = float(coef[j])

    def _read_sol(self):
        # solve and read the full solution + objective value
        _solve_model(self._solverfac, self._model)
        sol = np.fromiter((pe.value(self.x[j]) for j in range(self.problem.num_vars)), dtype=float)
        return sol, float(pe.value(self._model.obj))

    def _add_cut(self, coef, rhs):
        # add coef @ x <= rhs to a fresh copy
        new_model = self.copy()
        expr = sum(float(coef[j]) * new_model.x[j] for j in range(self.problem.num_vars)) <= float(
            rhs
        )
        new_model._model.cons.add(expr)
        return new_model

    def _build_flat_vars(self, m):
        # one indexed Var with per-entry domain and bounds
        prob = self.problem
        domain = {EPO.BINARY: pe.Binary, EPO.INTEGER: pe.Integers, EPO.CONTINUOUS: pe.Reals}

        def dom(_m, j):
            return domain[prob.var_type[j]]

        def bnd(_m, j):
            lo = None if np.isneginf(prob.var_lb[j]) else float(prob.var_lb[j])
            hi = None if np.isposinf(prob.var_ub[j]) else float(prob.var_ub[j])
            return lo, hi

        m.x = pe.Var(range(prob.num_vars), domain=dom, bounds=bnd)
        return m.x

    def _emit_constraints(self, m, x):
        # linear (Q is None) and quadratic constraints from the finalized IR
        m.cons = pe.ConstraintList()
        for Q, A, sense, b in self.problem.constrs:
            A = A.tocsr()
            b = np.asarray(b, dtype=float).reshape(-1)
            if Q is None:
                # one relation per row of A
                for r in range(A.shape[0]):
                    row = A.getrow(r)
                    lhs = sum(v * x[j] for j, v in zip(row.indices, row.data))
                    _add_relation(m.cons, lhs, sense, b[r])
            else:
                a = np.asarray(A.todense(), dtype=float).reshape(-1)
                lhs = _quad(Q, x)
                if a.any():
                    lhs = lhs + sum(a[j] * x[j] for j in np.nonzero(a)[0])
                _add_relation(m.cons, lhs, sense, b[0])
