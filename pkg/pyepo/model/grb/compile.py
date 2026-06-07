#!/usr/bin/env python
"""
Gurobi compiler for the PyEPO DSL.

``compiledGrbProblem`` mixes the generic ``compiledBase`` with ``optGrbModel``
to turn a finalized DSL ``Problem`` into a GurobiPy model. It builds the model
and provides the Gurobi read / write hooks; the objective handling lives in
``compiledBase``.
"""

from __future__ import annotations

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    pass

from pyepo import EPO
from pyepo.dsl.compiled import compiledBase
from pyepo.model.grb.grbmodel import optGrbModel


def compileProblem(problem, **params) -> compiledGrbProblem:
    """Instantiate the Gurobi-compiled problem; ``params`` are Gurobi parameters."""
    return compiledGrbProblem(problem, params=params)


class compiledGrbProblem(compiledBase, optGrbModel):
    """
    Gurobi-backed compiled DSL problem.
    """

    def _getModel(self) -> tuple:
        # build the gurobi model from the finalized IR
        prob = self.problem
        m = gp.Model()
        m.Params.outputFlag = 0
        # objective sense (EPO -> Gurobi)
        m.modelSense = GRB.MAXIMIZE if prob.objective.modelSense == EPO.MAXIMIZE else GRB.MINIMIZE
        x = self._build_flat_vars(m)
        self._emit_constraints(m, x)
        # parameter-free quadratic objective term
        if prob.obj_Q is not None:
            m.setObjective(x @ prob.obj_Q @ x)
        return m, x

    def _apply_params(self):
        # apply the Gurobi solver parameters
        for key, value in self.params.items():
            self._model.setParam(key, value)

    def _write_obj(self, coef):
        # set the full-length objective coefficient on the MVar
        self.x.Obj = coef

    def _read_sol(self):
        # optimize and read the full solution + objective value
        self._model.optimize()
        return np.asarray(self.x.x), self._model.objVal

    def _add_cut(self, coef, rhs):
        # add coef @ x <= rhs to a fresh copy
        new_model = self.copy()
        new_model._model.addConstr(coef @ new_model.x <= float(rhs))
        new_model._model.update()
        return new_model

    def _build_flat_vars(self, m):
        # one MVar with per-entry bounds and type
        prob = self.problem
        lb = np.where(np.isneginf(prob.var_lb), -GRB.INFINITY, prob.var_lb)
        ub = np.where(np.isposinf(prob.var_ub), GRB.INFINITY, prob.var_ub)
        # EPO type -> Gurobi vtype
        grb_vtype = {EPO.BINARY: GRB.BINARY, EPO.INTEGER: GRB.INTEGER, EPO.CONTINUOUS: GRB.CONTINUOUS}
        vtype = [grb_vtype[t] for t in prob.var_type]
        return m.addMVar(prob.num_vars, lb=lb, ub=ub, vtype=vtype, name=prob.cost_var.name or "x")

    def _emit_constraints(self, m, x):
        # linear (Q is None) or quadratic constraints from the finalized IR
        for i, (Q, A, sense, b) in enumerate(self.problem.constrs):
            if Q is None:
                expr = A @ x
                rhs = np.asarray(b, dtype=float)
            else:
                a = np.asarray(A.todense(), dtype=float).reshape(-1)
                expr = x @ Q @ x + (a @ x if a.any() else 0.0)
                rhs = float(np.asarray(b, dtype=float).reshape(-1)[0])
            # name each constraint group
            name = f"c{i}"
            if sense == "<=":
                m.addConstr(expr <= rhs, name=name)
            elif sense == ">=":
                m.addConstr(expr >= rhs, name=name)
            else:
                m.addConstr(expr == rhs, name=name)
