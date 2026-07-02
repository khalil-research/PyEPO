#!/usr/bin/env python
"""
Shared DSL compile hooks for MVar-style solver backends.

``compiledMVarMixin`` carries the model build and constraint emission common to
the Gurobi and COPT compilers, which expose the same matrix-variable algebra
(``addMVar``, ``A @ x``, ``x.Obj``). Backend subclasses supply the solver
handle (``_new_model``), the variable specs (``_var_specs``), and the
quadratic-constraint adder (``_quad_adder``).
"""

from __future__ import annotations

import numpy as np


class compiledMVarMixin:
    """
    Shared compile hooks for MVar-style backends (Gurobi / COPT).
    """

    def _getModel(self) -> tuple:
        # build the solver model from the finalized IR
        prob = self.problem
        m = self._new_model()
        x = self._build_flat_vars(m)
        self._emit_constraints(m, x)
        # parameter-free quadratic objective term
        if prob.obj_Q is not None:
            m.setObjective(x @ prob.obj_Q @ x)
        return m, x

    def _new_model(self):
        # backend hook: solver model with objective sense applied
        raise NotImplementedError

    def _var_specs(self) -> tuple[float, dict, str]:
        # backend hook: (infinity, EPO-to-backend vtype map, MVar name kwarg)
        raise NotImplementedError

    def _quad_adder(self, m):
        # backend hook: quadratic-constraint adder
        return m.addConstr

    def _apply_params(self):
        # apply solver params; the canonical `timelimit` (seconds) maps to TimeLimit
        for key, value in self.params.items():
            self._model.setParam("TimeLimit" if key == "timelimit" else key, value)

    def _write_obj(self, coef):
        # set the full-length objective coefficient on the MVar
        self.x.Obj = coef

    def _build_flat_vars(self, m):
        # one MVar with per-entry bounds and type
        prob = self.problem
        inf, vtype_map, name_kw = self._var_specs()
        lb = np.where(np.isneginf(prob.var_lb), -inf, prob.var_lb)
        ub = np.where(np.isposinf(prob.var_ub), inf, prob.var_ub)
        vtype = [vtype_map[t] for t in prob.var_type]
        name = prob.cost_var_name or "x"
        return m.addMVar(prob.num_vars, lb=lb, ub=ub, vtype=vtype, **{name_kw: name})

    def _emit_constraints(self, m, x):
        # linear (Q is None) or quadratic constraints from the finalized IR
        for i, (Q, A, sense, b) in enumerate(self.problem.constrs):
            if Q is None:
                expr = A @ x
                rhs = np.asarray(b, dtype=float)
                add = m.addConstr
            else:
                a = np.asarray(A.todense(), dtype=float).reshape(-1)
                expr = x @ Q @ x + (a @ x if a.any() else 0.0)
                rhs = float(np.asarray(b, dtype=float).reshape(-1)[0])
                add = self._quad_adder(m)
            # name each constraint group
            name = f"c{i}"
            if sense == "<=":
                add(expr <= rhs, name=name)
            elif sense == ">=":
                add(expr >= rhs, name=name)
            else:
                add(expr == rhs, name=name)
