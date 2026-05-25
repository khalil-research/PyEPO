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
from pyepo.model.opt import optModel
from pyepo.utils import costToNumpy

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


def _is_mvar(x) -> bool:
    return _CoptMVar is not None and isinstance(x, _CoptMVar)


_envr = None


def _get_envr():
    """Process-local COPT Envr singleton; avoids re-booting license/threads per model build."""
    global _envr
    if _envr is None:
        _envr = Envr()
    return _envr


class optCoptModel(optModel):
    """
    This is an abstract class for a Cardinal Optimizer optimization model

    Attributes:
        _model (COPT model): COPT model
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
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector does not match number of cost variables.")
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
        if _is_mvar(self.x):
            sol = np.asarray(self.x.x)
        else:
            sol = np.asarray(self._model.getInfo("Value", self._vars_list))
        return sol, self._model.objVal

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
            new_model.x = _CoptMVar.fromlist(new_vars)
            new_model._vars_list = None
        else:
            new_model.x = {key: new_vars[i] for i, key in enumerate(self.x)}
            new_model._vars_list = list(new_model.x.values())
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
        new_model = self.copy()
        coefs = costToNumpy(coefs)
        if _is_mvar(new_model.x):
            new_model._model.addConstr(coefs @ new_model.x <= rhs)
        else:
            # LinExpr().addTerms builds the affine expression in one C call
            expr = LinExpr()
            expr.addTerms(coefs.tolist(), new_model._vars_list)
            new_model._model.addConstr(expr <= rhs)
        return new_model
