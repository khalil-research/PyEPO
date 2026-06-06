#!/usr/bin/env python
"""
Gurobi compiler for the PyEPO DSL.

``compiledGrbProblem`` mixes the generic ``compiledBase`` with ``optGrbModel``
to turn a finalized DSL ``Problem`` into a GurobiPy model. It builds the flat
solver variables, emits the (linear / quadratic) constraints, records the
cost-paired ``_cost_vars``, and provides the cost read/write hooks
``compiledBase`` drives.
"""

from __future__ import annotations

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    pass

from pyepo.dsl.compiled import compiledBase
from pyepo.model.grb.grbmodel import optGrbModel
from pyepo.utils import costToNumpy


def compileProblem(problem, **kwargs) -> compiledGrbProblem:
    """Instantiate the Gurobi-compiled problem (dispatcher entry)."""
    return compiledGrbProblem(problem, **kwargs)


class compiledGrbProblem(compiledBase, optGrbModel):
    """
    Gurobi-backed compiled DSL problem (uniform-vtype MVar path).
    """

    def __init__(self, problem):
        # attr name matches the __init__ arg so getArgs rebuilds in a worker
        self.problem = problem
        super().__init__()

    def _getModel(self) -> tuple:
        # build a gurobi model from the finalized IR
        prob = self.problem
        m = gp.Model()
        m.Params.outputFlag = 0
        m.modelSense = GRB.MAXIMIZE if prob.objective.sense == "maximize" else GRB.MINIMIZE
        # variables
        x = self._build_flat_vars(m)
        # constraints
        self._emit_constraints(m, x)
        # parameter-free quadratic objective term (linear cost set later via setObj)
        if prob.obj_Q is not None:
            m.setObjective(x @ prob.obj_Q @ x)
        # cost-paired solver vars
        self._record_cost_layout(x)
        return m, x

    def _build_flat_vars(self, m):
        # uniform vtype -> a single MVar; mixed vtype (dict) is a later step
        prob = self.problem
        lb = np.where(np.isneginf(prob.var_lb), -GRB.INFINITY, prob.var_lb)
        ub = np.where(np.isposinf(prob.var_ub), GRB.INFINITY, prob.var_ub)
        if prob.var_binary.all():
            return m.addMVar(prob.n_total, vtype=GRB.BINARY, name="x")
        if prob.var_integer.all():
            return m.addMVar(prob.n_total, lb=lb, ub=ub, vtype=GRB.INTEGER, name="x")
        if not prob.var_binary.any() and not prob.var_integer.any():
            return m.addMVar(prob.n_total, lb=lb, ub=ub, name="x")
        raise NotImplementedError("mixed-vtype (dict-of-Var) compilation is a later Phase 2 step.")

    def _emit_constraints(self, m, x):
        # linear (Q is None) or quadratic constraints from the finalized IR
        for Q, A, sense, b in self.problem.constraints_ir:
            if Q is None:
                expr = A @ x
                rhs = np.asarray(b, dtype=float)
            else:
                a = np.asarray(A.todense(), dtype=float).reshape(-1)
                expr = x @ Q @ x + (a @ x if a.any() else 0.0)
                rhs = float(np.asarray(b, dtype=float).reshape(-1)[0])
            if sense == "<=":
                m.addConstr(expr <= rhs)
            elif sense == ">=":
                m.addConstr(expr >= rhs)
            else:
                m.addConstr(expr == rhs)

    def _record_cost_layout(self, x):
        # cost-paired solver vars in flat order (the same hook legacy TSP/VRP use)
        sl = self.problem.flat_slice[self.problem.cost_var]
        self._cost_vars = x[sl].tolist()

    # ---- cost read/write hooks driven by compiledBase ----

    def _writeObj(self, coef):
        # set the objective coefficient on the cost-paired vars
        self._model.setAttr("Obj", self._cost_vars, costToNumpy(coef).tolist())

    def _readCostSol(self):
        # optimize, then read the cost-paired solution
        self._model.optimize()
        vals = np.asarray(self._model.getAttr("X", self._cost_vars), dtype=np.float32)
        return vals, self._model.objVal

    def _writeConstr(self, coef, rhs):
        # add one linear constraint over the cost-paired vars to a fresh copy
        new_model = self.copy()
        expr = gp.LinExpr(np.asarray(coef, dtype=float).tolist(), new_model._cost_vars) <= float(rhs)
        new_model._model.addConstr(expr)
        new_model._model.update()
        return new_model

    def copy(self):
        # rebuild a fresh model from the immutable IR (re-creates _cost_vars)
        return type(self)(self.problem)
