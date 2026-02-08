#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on Pyomo
"""

from copy import copy

import numpy as np
import torch

from pyepo import EPO
from pyepo.model.opt import optModel

try:
    from pyomo import opt as po
    from pyomo import environ as pe
    _HAS_PYOMO = True
except ImportError:
    _HAS_PYOMO = False


class optOmoModel(optModel):
    """
    This is an abstract class for a Pyomo-based optimization model

    Attributes:
        _model (Pyomo model): Pyomo model
        solver (str): optimization solver in the background
    """

    def __init__(self, solver="glpk"):
        """
        Args:
            solver (str): optimization solver in the background
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

    def __repr__(self):
        return "optOmoModel " + self.__class__.__name__

    def setObj(self, c):
        """
        A method to set the objective function

        Args:
            c (np.ndarray / list): cost of objective function
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector cannot match vars.")
        # check if c is a PyTorch tensor
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        else:
            c = np.asarray(c, dtype=np.float32)
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

    def solve(self):
        """
        A method to solve the model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        # solve
        self._solverfac.solve(self._model)
        return [pe.value(self.x[k]) for k in self.x], pe.value(self._model.obj)

    def copy(self):
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

    def addConstr(self, coefs, rhs):
        """
        A method to add a new constraint

        Args:
            coefs (np.ndarray / list): coefficients of new constraint
            rhs (float): right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        if len(coefs) != self.num_cost:
            raise ValueError("Size of coef vector does not match number of cost variables.")
        # copy
        new_model = self.copy()
        # add constraint
        expr = sum(coefs[i] * new_model.x[k]
                   for i, k in enumerate(new_model.x)) <= rhs
        new_model._model.cons.add(expr)
        return new_model
