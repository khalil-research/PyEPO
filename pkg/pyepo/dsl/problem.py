#!/usr/bin/env python
"""
The DSL ``Problem``: collects variables, finalizes the symbolic IR into global
sparse matrices, validates PyEPO scope, and compiles to a backend.

At construction every referenced ``Variable`` is assigned a contiguous slice of
a global flat index (encounter order: objective first, then each constraint in
list order). Constraints finalize to ``(Q | None, A, sense, b_eff)`` and the
objective records ``cost_param`` / ``cost_var`` plus an optional parameter-free
quadratic offset. The DSL core targets 1:1 problems (``num_vars == num_cost``).
"""

from __future__ import annotations

import copy as _copy

import numpy as np

from pyepo import EPO


class Problem:
    """
    A symbolic predict-then-optimize problem.

    Attributes:
        objective (Minimize | Maximize): the objective
        constraints (list[Constraint]): the constraints
        cost_param (Parameter): the unique predicted-cost Parameter
        cost_var (Variable): the Variable the cost pairs with (1:1)
        num_cost (int): predicted dimension (``cost_param.size``)
        flat_slice (dict[Variable, slice]): global flat slice per Variable
        num_vars (int): total decision-variable count
        constrs (list[tuple]): finalized ``(Q, A, sense, b_eff)`` per constraint
        obj_Q (csr_matrix | None): finalized parameter-free quadratic objective term
    """

    def __init__(self, objective, constraints=None):
        self.objective = objective
        self.constraints = list(constraints) if constraints else []
        # PyEPO-scope checks
        self._validate()
        # objective cost layout
        self.cost_param = objective.cost_param
        self.cost_var = objective.cost_var
        # assign flat slices in encounter order (objective var first)
        self._assign_flat()
        # finalize IR
        self._finalize()

    @property
    def num_cost(self) -> int:
        # predicted cost dimension = the unique Parameter's size
        return self.cost_param.size

    def _assign_flat(self):
        # collect variables in encounter order; dedup by identity (Variable.__eq__ is overloaded)
        order, seen = [self.cost_var], {id(self.cost_var)}
        for con in self.constraints:
            for v in con.variables():
                if id(v) not in seen:
                    seen.add(id(v))
                    order.append(v)
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
        # objective is a Minimize / Maximize wrapping a 1:1 ParametricBilinear
        from pyepo.dsl.expression import ParametricBilinear
        from pyepo.dsl.objective import Objective
        if not isinstance(self.objective, Objective):
            raise TypeError("Problem objective must be a Minimize / Maximize.")
        obj = self.objective.expr
        if not isinstance(obj, ParametricBilinear):
            raise TypeError("Objective must be a ParametricBilinear (e.g. c @ x).")
        if obj.cost_param.size != obj.cost_var.size:
            raise ValueError(
                f"cost_param size {obj.cost_param.size} != cost_var size {obj.cost_var.size}."
            )
        # a quadratic objective term must be pure (no fixed linear / constant part)
        if obj.offset is not None:
            has_linear = any(b.nnz > 0 for b in obj.offset.affine.blocks.values())
            if has_linear or bool(obj.offset.affine.const.any()):
                raise ValueError(
                    "The objective's quadratic term must have no linear or constant part "
                    "(only the predicted cost is linear)."
                )

    def _finalize(self):
        # constraints -> global (Q, A, sense, b_eff)
        self.constrs = [con.finalize(self.flat_slice, self.num_vars) for con in self.constraints]
        # parameter-free quadratic objective offset (QP term), if any
        offset = self.objective.expr.offset
        self.obj_Q = None
        from pyepo.dsl.expression import Quadratic
        if isinstance(offset, Quadratic):
            self.obj_Q = offset.finalize_Q(self.flat_slice, self.num_vars)

    def relax(self) -> Problem:
        """
        Return a new Problem with all integer / binary variables continuous
        (bounds preserved: binary ⇒ [0, 1]); objective and constraints unchanged.
        """
        new = _copy.copy(self)
        new.var_type = np.full(self.num_vars, EPO.CONTINUOUS, dtype=object)
        return new

    def compile(self, backend, **kwargs):
        """
        Compile to a solver backend, returning an ``optModel``.

        Args:
            backend: solver backend name (currently ``"gurobi"``).
        """
        # route to the backend compiler
        if backend == "gurobi":
            from pyepo.model.grb.compile import compileProblem
            return compileProblem(self, **kwargs)
        raise NotImplementedError(f"DSL backend {backend!r} is not supported (available: 'gurobi').")
