#!/usr/bin/env python
"""
Abstract optimization model based on Google OR-Tools (pywraplp)
"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import numpy as np

try:
    from ortools.linear_solver import pywraplp

    _HAS_ORTOOLS = True
except ImportError:
    _HAS_ORTOOLS = False

from pyepo import EPO
from pyepo.model.opt import costToNumpy, optModel

if TYPE_CHECKING:
    import torch


class optOrtModel(optModel):
    """
    This is an abstract class for an OR-Tools pywraplp optimization model

    Attributes:
        _model (pywraplp.Solver): OR-Tools linear solver
        solver (str): solver backend (e.g. "scip", "glop", "cbc")
    """

    def __init__(self, solver: str = "scip") -> None:
        """
        Args:
            solver: solver backend for pywraplp
        """
        if not _HAS_ORTOOLS:
            raise ImportError(
                "OR-Tools is not installed. Please install ortools to use this feature."
            )
        self.solver = solver
        self._extra_constrs = []
        super().__init__()
        # suppress output
        self._model.SuppressOutput()

    def __repr__(self) -> str:
        return "optOrtModel " + self.__class__.__name__

    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        """
        A method to set the objective function

        Args:
            c: cost of objective function
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector does not match number of cost variables.")
        c = costToNumpy(c)
        # set obj
        obj = self._model.Objective()
        obj.Clear()
        for i, k in enumerate(self.x):
            obj.SetCoefficient(self.x[k], float(c[i]))
        if self.modelSense == EPO.MAXIMIZE:
            obj.SetMaximization()
        else:
            obj.SetMinimization()

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve the model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        status = self._model.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            raise RuntimeError(f"Solver did not find an optimal solution. Status: {status}")
        sol = np.array([self.x[k].solution_value() for k in self.x], dtype=np.float32)
        obj = self._model.Objective().Value()
        return sol, obj

    def copy(self) -> optOrtModel:
        """
        A method to copy the model

        Returns:
            optModel: new copied model
        """
        new_model = copy(self)
        new_model._extra_constrs = list(self._extra_constrs)
        # rebuild model from scratch
        new_model._model, new_model.x = new_model._getModel()
        new_model._model.SuppressOutput()
        # replay extra constraints
        for coefs, rhs in new_model._extra_constrs:
            ct = new_model._model.Constraint(-new_model._model.infinity(), float(rhs))
            for i, k in enumerate(new_model.x):
                ct.SetCoefficient(new_model.x[k], float(coefs[i]))
        return new_model

    def addConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> optOrtModel:
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
        # store and add constraint
        new_model._extra_constrs.append((list(coefs), float(rhs)))
        ct = new_model._model.Constraint(-new_model._model.infinity(), float(rhs))
        for i, k in enumerate(new_model.x):
            ct.SetCoefficient(new_model.x[k], float(coefs[i]))
        return new_model


if __name__ == "__main__":
    import random

    # random seed
    random.seed(42)
    np.random.seed(42)
    # number of variables
    num_vars = 10
    # create a simple LP model for testing
    solver = pywraplp.Solver.CreateSolver("GLOP")
    x = {i: solver.NumVar(0, 10, f"x_{i}") for i in range(num_vars)}
    for i in range(num_vars):
        solver.Add(x[i] <= 5)
    solver.Maximize(sum(x[i] for i in range(num_vars)))
    status = solver.Solve()
    print(f"Status: {status}")
    print(f"Objective: {solver.Objective().Value()}")
    for i in range(num_vars):
        print(f"x_{i} = {x[i].solution_value()}")
