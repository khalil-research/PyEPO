#!/usr/bin/env python
"""
The DSL ``Problem``: collects variables, finalizes the symbolic IR into global
sparse matrices, validates PyEPO scope, and (Phase 2) compiles to a backend.

At construction every referenced ``Variable`` is assigned a contiguous slice of
a global flat index (encounter order: objective first, then each constraint in
list order). Constraints finalize to ``(Q | None, A, sense, b_eff)`` and the
objective records ``cost_param`` / ``cost_var`` / ``cost_gather`` plus an
optional parameter-free quadratic offset.
"""

from __future__ import annotations

import copy as _copy

import numpy as np

from pyepo.dsl.validation import validate


class Problem:
    """
    A symbolic predict-then-optimize problem.

    Attributes:
        objective (Minimize | Maximize): the objective
        constraints (list[Constraint]): the constraints
        cost_param (Parameter): the unique predicted-cost Parameter
        cost_var (Variable): the Variable the cost pairs with
        cost_gather (np.ndarray | None): cost index per cost_var position; None ⇒ 1:1
        num_cost (int): predicted dimension (``cost_param.size``)
        flat_slice (dict[Variable, slice]): global flat slice per Variable
        n_total (int): total flat variable count
        constraints_ir (list[tuple]): finalized ``(Q, A, sense, b_eff)`` per constraint
        obj_Q (csr_matrix | None): finalized parameter-free quadratic objective term
    """

    def __init__(self, objective, constraints=None):
        self.objective = objective
        self.constraints = list(constraints) if constraints else []
        # objective cost layout
        self.cost_param = objective.cost_param
        self.cost_var = objective.cost_var
        self.cost_gather = objective.expr.cost_gather
        # assign flat slices in encounter order (objective var first)
        self._assign_flat()
        # PyEPO-scope checks
        validate(self)
        # finalize IR
        self._finalize()

    @property
    def num_cost(self) -> int:
        # predicted cost dimension = the unique Parameter's size
        return self.cost_param.size

    def _assign_flat(self):
        # collect variables in encounter order, assign contiguous flat slices
        order = [self.cost_var]
        for con in self.constraints:
            for v in con.variables():
                if v not in order:
                    order.append(v)
        self.variables = order
        self.flat_slice = {}
        n = 0
        for v in order:
            self.flat_slice[v] = slice(n, n + v.size)
            n += v.size
        self.n_total = n
        # flat-space bounds and types
        self.var_lb = np.concatenate([v.lb for v in order]) if order else np.zeros(0)
        self.var_ub = np.concatenate([v.ub for v in order]) if order else np.zeros(0)
        self.var_binary = np.concatenate([np.full(v.size, v.binary) for v in order]).astype(bool)
        self.var_integer = np.concatenate([np.full(v.size, v.integer) for v in order]).astype(bool)

    def _finalize(self):
        # constraints -> global (Q, A, sense, b_eff)
        self.constraints_ir = [con.finalize(self.flat_slice, self.n_total) for con in self.constraints]
        # parameter-free quadratic objective offset (QP term), if any
        offset = self.objective.expr.offset
        self.obj_Q = None
        from pyepo.dsl.expression import Quadratic
        if isinstance(offset, Quadratic):
            self.obj_Q = offset.finalize_Q(self.flat_slice, self.n_total)

    def relax(self) -> Problem:
        """
        Return a new Problem with all integer / binary variables continuous
        (bounds preserved: binary ⇒ [0, 1]); objective and constraints unchanged.
        """
        new = _copy.copy(self)
        new.var_binary = np.zeros_like(self.var_binary)
        new.var_integer = np.zeros_like(self.var_integer)
        return new

    def compile(self, backend, **kwargs):
        """
        Compile to a solver backend. Backend compilers are Phase 2.
        """
        raise NotImplementedError(
            "DSL backend compilers are not implemented yet (Phase 2); "
            "the Phase 1 core builds and finalizes the IR only."
        )
