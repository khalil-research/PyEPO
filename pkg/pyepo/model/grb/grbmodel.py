#!/usr/bin/env python
"""
Abstract optimization model based on GurobiPy
"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB

    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

from pyepo import EPO
from pyepo.model._common import validate_constraint, validate_objective_shape
from pyepo.model.opt import optModel
from pyepo.utils import costToNumpy

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


def _require_solution(model) -> None:
    """Raise a stable error when Gurobi did not produce a usable solution."""
    if model.SolCount == 0:
        raise RuntimeError(f"Gurobi found no solution (status {model.Status}).")


class optGrbModel(optModel):
    """
    Abstract base class for GurobiPy-backed optimization models.

    Subclasses implement ``_getModel`` to build a ``gurobipy.Model`` and
    return ``(model, variables)``. The objective sense is auto-detected
    from the underlying Gurobi model -- ``modelSense`` does not need to be
    set manually. Solver output is silenced by default; cost-vector updates
    use the C-level ``setAttr("Obj", ...)`` batch path for efficiency.

    Attributes:
        _model (gurobipy.Model): underlying Gurobi model
    """

    def __init__(self) -> None:
        # error
        if not _HAS_GUROBI:
            raise ImportError(
                "Gurobi is not installed. Please install gurobipy to use this feature."
            )
        super().__init__()
        # model sense
        self._model.update()
        if self._model.modelSense == GRB.MINIMIZE:
            self.modelSense = EPO.MINIMIZE
        elif self._model.modelSense == GRB.MAXIMIZE:
            self.modelSense = EPO.MAXIMIZE
        else:
            raise ValueError("Invalid modelSense.")
        # turn off output
        self._model.Params.outputFlag = 0
        # cache ordered Var list once for setAttr/getAttr batch paths
        self._vars_list: list[gp.Var] | None = (
            None if isinstance(self.x, gp.MVar) else list(self.x.values())
        )

    def __repr__(self) -> str:
        return "optGRBModel " + self.__class__.__name__

    @property
    def num_cost(self) -> int:
        """
        number of costs to be predicted
        """
        return self.x.size if isinstance(self.x, gp.MVar) else len(self.x)

    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        """
        A method to set the objective function

        Args:
            c: cost of objective function
        """
        validate_objective_shape(c, self.num_cost)
        c = costToNumpy(c)
        if isinstance(self.x, gp.MVar):
            # direct Obj attr write skips MLinExpr allocation
            self.x.Obj = c  # type: ignore[attr-defined]
        else:
            # batch C-level coefficient update
            self._model.setAttr("Obj", self._vars_list, c.tolist())

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve the model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        # optimize() flushes pending changes
        self._model.optimize()
        _require_solution(self._model)
        if isinstance(self.x, gp.MVar):
            sol = self.x.x  # type: ignore[attr-defined]
        else:
            sol = np.asarray(self._model.getAttr("X", self._vars_list))
        obj = self._model.objVal
        return sol, obj

    def copy(self) -> Self:
        """
        A method to copy the model

        Returns:
            optModel: new copied model
        """
        new_model = copy(self)
        # update model
        self._model.update()
        # new model
        new_model._model = self._model.copy()
        # variables for new model (preserve MVar type if original uses MVar)
        new_vars = new_model._model.getVars()
        if isinstance(self.x, gp.MVar):
            new_model.x = gp.MVar.fromlist(new_vars)
            new_model._vars_list = None
        else:
            # map by var name so auxiliary vars or ordering can't misalign the dict
            new_by_name = {v.VarName: v for v in new_vars}
            new_model.x = {key: new_by_name[var.VarName] for key, var in self.x.items()}
            new_model._vars_list = list(new_model.x.values())
        return new_model

    def _copy_objective_to(self, other: optGrbModel) -> None:
        """Copy Gurobi objective coefficients aligned with predicted costs."""
        self._model.update()
        if self._cost_vars:
            coefs = np.asarray(self._model.getAttr("Obj", self._cost_vars))
            if coefs.size != self.num_cost:
                coefs = coefs.reshape(self.num_cost, -1)[:, 0]
        elif isinstance(self.x, gp.MVar):
            coefs = np.asarray(self.x.Obj)  # type: ignore[attr-defined]
        else:
            coefs = np.asarray(self._model.getAttr("Obj", self._vars_list))
        other.setObj(coefs)

    def addConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> Self:
        """
        A method to add a new constraint

        Args:
            coefs: coefficients of new constraint
            rhs: right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        rhs = validate_constraint(coefs, rhs, self.num_cost)
        coefs_np: np.ndarray = costToNumpy(coefs).copy()
        # copy
        new_model = self.copy()
        # add constraint
        if isinstance(new_model.x, gp.MVar):
            new_model._model.addConstr(coefs_np @ new_model.x <= rhs)
        else:
            # LinExpr(coeffs, vars) builds the affine expression in one C call
            vars_list = new_model._vars_list
            assert vars_list is not None
            expr = gp.LinExpr(coefs_np.tolist(), vars_list) <= rhs
            new_model._model.addConstr(expr)
        # track for replay on relax
        new_model._extra_constrs = [*self._extra_constrs, (coefs_np, rhs)]
        return new_model


def _lazy_cut_key(tc) -> tuple | None:
    """Structural identity of a TempConstr: sorted (var, coef) terms with sense and rhs."""
    lhs = getattr(tc, "_lhs", None)
    rhs = getattr(tc, "_rhs", None)
    sense = getattr(tc, "_sense", None)
    if lhs is None or rhs is None or sense is None:
        return None
    terms = tuple(
        sorted((lhs.getVar(i).VarName, round(lhs.getCoeff(i), 9)) for i in range(lhs.size()))
    )
    return (terms, sense, round(float(rhs), 9))


def _promote_lazy_cuts(grb_model, recycled_keys: set) -> None:
    """Move the previous solve's tracked lazy cuts into the model's lazy pool (``Constr.Lazy = 1``)."""
    new_constrs = []
    for tc in getattr(grb_model, "_lazy_constrs", []):
        key = _lazy_cut_key(tc)
        if key is None or key in recycled_keys:
            continue
        recycled_keys.add(key)
        new_constrs.append(grb_model.addConstr(tc))
    if new_constrs:
        grb_model.update()
        for constr in new_constrs:
            constr.Lazy = 1
