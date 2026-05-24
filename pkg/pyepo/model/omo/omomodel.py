#!/usr/bin/env python
"""
Abstract optimization model based on Pyomo
"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import numpy as np

from pyepo import EPO
from pyepo.model.opt import optModel
from pyepo.utils import costToNumpy

try:
    from pyomo import environ as pe
    from pyomo import opt as po

    _HAS_PYOMO = True
except ImportError:
    _HAS_PYOMO = False

if TYPE_CHECKING:
    import torch


class optOmoModel(optModel):
    """
    This is an abstract class for a Pyomo-based optimization model

    Attributes:
        _model (Pyomo model): Pyomo model
        solver (str): optimization solver in the background
    """

    def __init__(self, solver: str = "glpk") -> None:
        """
        Args:
            solver: optimization solver in the background
        """
        # error
        if not _HAS_PYOMO:
            raise ImportError("Pyomo is not installed. Please install pyomo to use this feature.")
        super().__init__()
        # init obj
        if self.modelSense == EPO.MINIMIZE:
            self._model.obj = pe.Objective(sense=pe.minimize, expr=0)
        elif self.modelSense == EPO.MAXIMIZE:
            self._model.obj = pe.Objective(sense=pe.maximize, expr=0)
        else:
            raise ValueError("Invalid modelSense.")
        # set solver
        self.solver = solver
        if self.solver == "gurobi":
            self._solverfac = po.SolverFactory(self.solver, solver_io="python")
        else:
            self._solverfac = po.SolverFactory(self.solver)

    def __repr__(self) -> str:
        return "optOmoModel " + self.__class__.__name__

    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        """
        A method to set the objective function

        Args:
            c: cost of objective function
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector does not match number of cost variables.")
        c = costToNumpy(c)
        # delete previous component
        self._model.del_component(self._model.obj)
        # set obj
        obj = sum(c[i] * self.x[k] for i, k in enumerate(self.x))
        if self.modelSense == EPO.MINIMIZE:
            self._model.obj = pe.Objective(sense=pe.minimize, expr=obj)
        elif self.modelSense == EPO.MAXIMIZE:
            self._model.obj = pe.Objective(sense=pe.maximize, expr=obj)
        else:
            raise ValueError("Invalid modelSense.")

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve the model

        Returns:
            tuple: optimal solution (np.ndarray) and objective value (float)
        """
        self._solverfac.solve(self._model)
        sol = np.fromiter(
            (pe.value(self.x[k]) for k in self.x),
            dtype=np.float32,
        )
        return sol, pe.value(self._model.obj)

    def copy(self) -> optOmoModel:
        """
        A method to copy the model

        Returns:
            optModel: new copied model
        """
        new_model = copy(self)
        # new model
        new_model._model = self._model.clone()
        # variables for new model
        new_model.x = new_model._model.x
        return new_model

    def addConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> optOmoModel:
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
        # copy
        new_model = self.copy()
        # add constraint
        expr = sum(coefs[i] * new_model.x[k] for i, k in enumerate(new_model.x)) <= rhs
        new_model._model.cons.add(expr)
        return new_model
