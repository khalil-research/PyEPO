#!/usr/bin/env python
"""
Abstract optimization model based on Cardinal Optimizer (COPT)
"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import numpy as np

try:
    from coptpy import COPT, Envr, LinExpr

    try:
        from coptpy import MVar as _CoptMVar
    except ImportError:
        _CoptMVar = None
    _HAS_COPT = True
except ImportError:
    _HAS_COPT = False
    _CoptMVar = None

from pyepo import EPO
from pyepo.model._common import validate_constraint, validate_objective_shape
from pyepo.model.opt import optModel
from pyepo.utils import costToNumpy

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeVar

    import torch
    from typing_extensions import Self

    _T = TypeVar("_T")


def _is_mvar(x) -> bool:
    return _CoptMVar is not None and isinstance(x, _CoptMVar)


def _read_solution(model, reader: Callable[[], _T]) -> tuple[_T, float]:
    """Read a COPT solution through one stable no-solution error boundary."""
    try:
        return reader(), model.objVal
    except Exception as e:  # coptpy raises generic errors on no-solution
        raise RuntimeError(f"COPT found no solution (status {model.status}).") from e


_envr = None


def _get_envr():
    """Process-local COPT Envr singleton; avoids re-booting license/threads per model build."""
    global _envr
    if _envr is None:
        _envr = Envr()
    return _envr


class optCoptModel(optModel):
    """
    Abstract base class for Cardinal Optimizer (COPT) backed models.

    Subclasses implement ``_getModel`` to build a COPT ``Model`` and return
    ``(model, variables)``. The objective sense is auto-detected from the
    underlying COPT model (no need to set ``modelSense`` manually); solver
    logging is silenced by default. Cost-vector updates use the C-level
    ``setInfo("Obj", ...)`` batch path for efficiency. A process-local COPT
    ``Envr`` singleton is reused across instances to avoid re-booting the
    license and worker threads on every model build.

    Attributes:
        _model (coptpy.Model): underlying COPT model
    """

    def __init__(self) -> None:
        if not _HAS_COPT:
            raise ImportError("COPT is not installed. Please install coptpy to use this feature.")
        super().__init__()
        if self._model.ObjSense == COPT.MINIMIZE:
            self.modelSense = EPO.MINIMIZE
        elif self._model.ObjSense == COPT.MAXIMIZE:
            self.modelSense = EPO.MAXIMIZE
        else:
            raise ValueError("Invalid modelSense.")
        self._model.setParam("Logging", 0)
        # cache ordered Var list for setInfo/getInfo batch paths
        self._vars_list = None if _is_mvar(self.x) else list(self.x.values())

    def __repr__(self) -> str:
        return "optCoptModel " + self.__class__.__name__

    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        """
        A method to set the objective function

        Args:
            c: cost of objective function
        """
        validate_objective_shape(c, self.num_cost)
        c = costToNumpy(c)
        if _is_mvar(self.x):
            # direct Obj attr write skips MLinExpr allocation
            self.x.Obj = c
        else:
            # batch C-level coefficient update
            self._model.setInfo("Obj", self._vars_list, c.tolist())

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve the model

        Returns:
            tuple: optimal solution and objective value
        """
        self._model.solve()

        def read():
            if _is_mvar(self.x):
                # MVar.x is a coptpy NdArray; tolist() flattens it to plain floats
                return np.asarray(self.x.x.tolist())
            return np.asarray(self._model.getInfo("Value", self._vars_list))

        return _read_solution(self._model, read)

    def copy(self) -> Self:
        """
        A method to copy the model

        Returns:
            optModel: new copied model
        """
        new_model = copy(self)
        new_model._model = self._model.clone()
        new_vars = new_model._model.getVars()
        if _is_mvar(self.x):
            new_model.x = _CoptMVar.fromlist(new_vars)  # type: ignore[union-attr]
            new_model._vars_list = None
        else:
            new_model.x = {key: new_vars[i] for i, key in enumerate(self.x)}
            new_model._vars_list = list(new_model.x.values())
        return new_model

    def _copy_objective_to(self, other: optCoptModel) -> None:
        """Copy COPT objective coefficients aligned with predicted costs."""
        if self._cost_vars:
            coefs = np.asarray(self._model.getInfo("Obj", self._cost_vars))
            if coefs.size != self.num_cost:
                coefs = coefs.reshape(self.num_cost, -1)[:, 0]
        elif _is_mvar(self.x):
            coefs = np.asarray(self.x.Obj.tolist())
        else:
            coefs = np.asarray(self._model.getInfo("Obj", self._vars_list))
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
        new_model = self.copy()
        coefs = costToNumpy(coefs)
        if _is_mvar(new_model.x):
            new_model._model.addConstr(coefs @ new_model.x <= rhs)
        else:
            # LinExpr().addTerms builds the affine expression in one C call
            expr = LinExpr()
            expr.addTerms(new_model._vars_list, coefs.tolist())
            new_model._model.addConstr(expr <= rhs)
        # track for replay on relax
        new_model._extra_constrs = [*self._extra_constrs, (coefs, rhs)]
        return new_model
