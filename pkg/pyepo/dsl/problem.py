#!/usr/bin/env python
"""
The DSL ``Problem``: collects variables, finalizes the symbolic IR into global
sparse matrices, validates PyEPO scope, and compiles to a backend.

At construction every referenced ``Variable`` is assigned a contiguous slice of
a global flat index (encounter order: objective first, then each constraint in
list order). Constraints finalize to ``(Q | None, A, sense, b_eff)``; the
objective finalizes to the predicted ``c_pred_index``, a known ``fixed_cost``
linear part, and an optional parameter-free quadratic term ``obj_Q``.
"""

from __future__ import annotations

import copy as _copy

import numpy as np

from pyepo import EPO


def _dense_row(A):
    # a (1, n) sparse row -> flat (n,) array
    return np.asarray(A.todense(), dtype=float).reshape(-1)


class Problem:
    """
    A symbolic predict-then-optimize problem.

    Attributes:
        objective (Minimize | Maximize): the objective
        constraints (list[Constraint]): the constraints
        cost_param (Parameter): the unique predicted-cost Parameter
        cost_var (Variable): the Variable the predicted cost lands on
        num_cost (int): predicted dimension (``cost_param.size``)
        flat_slice (dict[Variable, slice]): global flat slice per Variable
        num_vars (int): total decision-variable count
        c_pred_index (np.ndarray): flat indices the predicted cost lands on
        fixed_cost (np.ndarray): known linear objective coefficients ``(num_vars,)``
        constrs (list[tuple]): finalized ``(Q, A, sense, b_eff)`` per constraint
        obj_Q (csr_matrix | None): finalized parameter-free quadratic objective term
    """

    def __init__(self, objective, constraints=None):
        from pyepo.dsl.expression import Constraint

        # a single bare constraint would hit Constraint.__bool__ via `if constraints`
        if isinstance(constraints, Constraint):
            raise TypeError("`constraints` must be a list; wrap it: [your_constraint].")
        self.objective = objective
        self.constraints = list(constraints) if constraints else []
        # every element must be a Constraint built with <= / >= / ==
        for i, con in enumerate(self.constraints):
            if not isinstance(con, Constraint):
                raise TypeError(
                    f"constraints[{i}] is {type(con).__name__}, not a Constraint; "
                    "build one with <=, >= or ==."
                )
        # PyEPO-scope checks
        self._validate()
        # objective cost layout
        self.cost_param = objective.cost_param
        self.cost_var = objective.cost_var
        self.modelSense = objective.modelSense
        self.cost_var_name = objective.cost_var.name
        # assign flat slices in encounter order (objective var first)
        self._assign_flat()
        # finalize IR
        self._finalize()

    def __repr__(self) -> str:
        # one-line summary of the finalized problem
        sense = "min" if self.modelSense == EPO.MINIMIZE else "max"
        n_quad = sum(1 for Q, *_ in self.constrs if Q is not None)
        quad = f" [{n_quad} quad]" if n_quad else ""
        obj_q = " +quad obj" if self.obj_Q is not None else ""
        return (
            f"Problem({sense}, {self.num_vars} vars, {len(self.constrs)} constrs"
            f"{quad}, cost dim={self.num_cost}{obj_q})"
        )

    @property
    def num_cost(self) -> int:
        # predicted cost dimension = the unique Parameter's size
        return self.cost_param.size

    def _assign_flat(self):
        # collect variables in encounter order; dedup by identity (Variable.__eq__ is overloaded)
        order, seen = [self.cost_var], {id(self.cost_var)}

        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                order.append(v)

        # objective-only variables: known fixed linear and quadratic terms
        expr = self.objective.expr
        if expr.fixed is not None:
            for v in expr.fixed.blocks:
                visit(v)
        if expr.quad is not None:
            for vi, vj in expr.quad.quad:
                visit(vi)
                visit(vj)
            for v in expr.quad.affine.blocks:
                visit(v)
        for con in self.constraints:
            for v in con.variables():
                visit(v)
        self.variables = order
        self.flat_slice = {}
        n = 0
        for v in order:
            self.flat_slice[v] = slice(n, n + v.size)
            n += v.size
        self.num_vars = n
        # flat-space bounds and types (order always holds at least cost_var)
        self.var_lb = np.concatenate([v.lb for v in order])
        self.var_ub = np.concatenate([v.ub for v in order])
        self.var_type = np.concatenate([v.vtype for v in order])

    def _validate(self):
        # objective is a Minimize / Maximize carrying a predicted cost term
        from pyepo.dsl.expression import ParametricObjective
        from pyepo.dsl.objective import Objective

        if not isinstance(self.objective, Objective):
            raise TypeError("Problem objective must be wrapped in Minimize(...) or Maximize(...).")
        if not isinstance(self.objective.expr, ParametricObjective):
            raise TypeError("Objective must include a predicted cost term (e.g. c @ x).")

    def _finalize(self):
        from pyepo.dsl.expression import Quadratic

        expr = self.objective.expr
        # predicted positions: the global flat indices the cost lands on
        sl = self.flat_slice[expr.cost_var]
        if expr.local_idx is None:
            self.c_pred_index = np.arange(sl.start, sl.stop)
        else:
            self.c_pred_index = sl.start + np.asarray(expr.local_idx, dtype=int)
        # known fixed linear coefficients over all variables; bare constants shift the objective value
        self.fixed_cost = np.zeros(self.num_vars)
        self.obj_offset = 0.0
        if expr.base is not None:
            self.fixed_cost[self.c_pred_index] += expr.base
        if expr.fixed is not None:
            self.fixed_cost += _dense_row(expr.fixed.finalize(self.flat_slice, self.num_vars))
            self.obj_offset += float(expr.fixed.const.sum())
        # parameter-free quadratic term; its linear part folds into fixed_cost
        self.obj_Q = None
        if isinstance(expr.quad, Quadratic):
            self.obj_Q = expr.quad.finalize_Q(self.flat_slice, self.num_vars)
            self.fixed_cost += _dense_row(expr.quad.affine.finalize(self.flat_slice, self.num_vars))
            self.obj_offset += float(expr.quad.affine.const.sum())
        # constraints -> global (Q, A, sense, b_eff)
        self.constrs = [con.finalize(self.flat_slice, self.num_vars) for con in self.constraints]

    def relax(self) -> Problem:
        """
        Return a new Problem with all integer / binary variables continuous
        (bounds preserved: binary ⇒ [0, 1]); objective and constraints unchanged.
        """
        new = _copy.deepcopy(self)
        new.var_type = np.full(self.num_vars, EPO.CONTINUOUS, dtype=object)
        return new

    def compile(self, backend, **kwargs):
        """
        Compile to a solver backend, returning an ``optModel``.

        A solver with a native backend (``"gurobi"`` / ``"copt"``) is reached
        through it directly — the most efficient path. The generic backends
        (``"pyomo"``, ``"ortools"``) are only for solvers without a native one
        (HiGHS, GLPK, CBC, SCIP, Ipopt); routing a native solver through them is
        wasteful indirection.

        Args:
            backend: solver backend name (``"gurobi"``, ``"copt"``, ``"pyomo"``, ``"ortools"``, ``"mpax"``).
            **kwargs: backend options -- ``solver=`` (generic backends), ``timelimit=`` (seconds), or native solver parameters.
        """
        # route to the backend compiler
        if backend == "gurobi":
            from pyepo.model.grb.compile import compileProblem

            return compileProblem(self, **kwargs)
        if backend == "copt":
            from pyepo.model.copt.compile import compileProblem

            return compileProblem(self, **kwargs)
        if backend == "pyomo":
            from pyepo.model.omo.compile import compileProblem

            return compileProblem(self, **kwargs)
        if backend == "ortools":
            from pyepo.model.ort.compile import compileProblem

            return compileProblem(self, **kwargs)
        if backend == "mpax":
            from pyepo.model.mpax.compile import compileProblem

            return compileProblem(self, **kwargs)
        raise NotImplementedError(
            f"DSL backend {backend!r} is not supported "
            "(available: 'gurobi', 'copt', 'pyomo', 'ortools', 'mpax')."
        )
