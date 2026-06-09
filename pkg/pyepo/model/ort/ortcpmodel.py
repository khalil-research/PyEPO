#!/usr/bin/env python
"""
Abstract optimization model based on Google OR-Tools CP-SAT
"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import numpy as np

try:
    from ortools.sat.python import cp_model

    _HAS_ORTOOLS = True
except ImportError:
    _HAS_ORTOOLS = False

from pyepo import EPO
from pyepo.model.opt import optModel
from pyepo.utils import costToNumpy

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


class optOrtCpModel(optModel):
    """
    Abstract base class for OR-Tools CP-SAT (constraint programming) models.

    Subclasses implement ``_getModel`` to build a ``cp_model.CpModel`` and
    return ``(model, variables)``. CP-SAT is an **integer-only** solver, so
    float cost vectors are scaled internally (multiplied by ``_OBJ_SCALE``
    and cast to int) before being passed to the solver; the objective value
    returned by ``solve`` is rescaled back to the original units.

    As with the other non-Gurobi/non-COPT backends, ``modelSense`` is not
    auto-detected -- set ``self.modelSense = EPO.MAXIMIZE`` in ``_getModel``
    for maximization (default is minimization). CP-SAT does not support LP
    relaxation, so ``relax()`` raises ``RuntimeError``.

    Attributes:
        _model (cp_model.CpModel): underlying OR-Tools CP-SAT model
    """

    _OBJ_SCALE = 1_000_000

    def __init__(self) -> None:
        if not _HAS_ORTOOLS:
            raise ImportError(
                "OR-Tools is not installed. Please install ortools to use this feature."
            )
        self._extra_constrs = []
        super().__init__()
        # cache ordered Var list for batched weighted_sum / per-Var Value() loop
        self._vars_list = list(self.x.values())

    def __repr__(self) -> str:
        return "optOrtCpModel " + self.__class__.__name__

    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        """
        A method to set the objective function

        Args:
            c: cost of objective function
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector does not match number of cost variables.")
        c = costToNumpy(c, dtype=np.float64)
        # scale float to int (round to match addConstr; truncation biases the objective)
        scaled = np.round(c * self._OBJ_SCALE).astype(np.int64).tolist()
        # C-level weighted sum avoids Python expression building per var
        obj_expr = cp_model.LinearExpr.weighted_sum(self._vars_list, scaled)
        if self.modelSense == EPO.MAXIMIZE:
            self._model.Maximize(obj_expr)
        else:
            self._model.Minimize(obj_expr)

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve the model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        solver = cp_model.CpSolver()
        status = solver.Solve(self._model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"Solver did not find a solution. Status: {solver.StatusName(status)}"
            )
        sol = np.fromiter(
            (solver.Value(v) for v in self._vars_list),
            dtype=np.float32,
            count=self.num_cost,
        )
        obj = solver.ObjectiveValue() / self._OBJ_SCALE
        return sol, obj

    def copy(self) -> Self:
        """
        A method to copy the model

        Returns:
            optModel: new copied model
        """
        new_model = copy(self)
        new_model._extra_constrs = list(self._extra_constrs)
        # rebuild model from scratch
        new_model._model, new_model.x = new_model._getModel()
        new_model._vars_list = list(new_model.x.values())
        # replay extra constraints
        for coefs, rhs in new_model._extra_constrs:
            lhs = cp_model.LinearExpr.weighted_sum(new_model._vars_list, [int(c) for c in coefs])
            new_model._model.Add(lhs <= int(rhs))
        return new_model

    def addConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> Self:
        """
        A method to add a new constraint

        Args:
            coefs: coefficients of new constraint
            rhs: right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        if len(coefs) != self.num_cost:
            raise ValueError("Size of coef vector does not match number of cost variables.")
        # scale to int
        scale = self._OBJ_SCALE
        scaled_coefs = [round(float(c) * scale) for c in coefs]
        scaled_rhs = round(float(rhs) * scale)
        # copy
        new_model = self.copy()
        # store and add constraint
        new_model._extra_constrs.append((scaled_coefs, scaled_rhs))
        new_model._model.Add(
            sum(scaled_coefs[i] * new_model.x[k] for i, k in enumerate(new_model.x)) <= scaled_rhs
        )
        return new_model

    def relax(self) -> optOrtCpModel:
        """
        CP-SAT does not support LP relaxation.
        """
        raise RuntimeError("CP-SAT does not support LP relaxation.")
