#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on GurobiPy
"""

from copy import copy

import numpy as np
import torch

try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

from pyepo import EPO
from pyepo.model.opt import optModel


class optGrbModel(optModel):
    """
    This is an abstract class for Gurobi-based optimization model

    Attributes:
        _model (GurobiPy model): Gurobi model
    """

    def __init__(self):
        super().__init__()
        # error
        if not _HAS_GUROBI:
            raise ImportError("Gurobi is not installed. Please install gurobipy to use this feature.")
        # model sense
        self._model.update()
        if self._model.modelSense == GRB.MINIMIZE:
            self.modelSense = EPO.MINIMIZE
        if self._model.modelSense == GRB.MAXIMIZE:
            self.modelSense = EPO.MAXIMIZE
        # turn off output
        self._model.Params.outputFlag = 0

    def __repr__(self):
        return "optGRBModel " + self.__class__.__name__

    @property
    def num_cost(self):
        """
        number of cost to be predicted
        """
        return self.x.size if isinstance(self.x, gp.MVar) else len(self.x)

    def setObj(self, c):
        """
        A method to set objective function

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
        # mvar
        if isinstance(self.x, gp.MVar):
            obj = c @ self.x
        # vars
        else:
            obj = gp.quicksum(c[i] * self.x[k] for i, k in enumerate(self.x))
        self._model.setObjective(obj)

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._model.update()
        self._model.optimize()
        # solution
        if isinstance(self.x, gp.MVar):
            sol = self.x.x
        else:
            sol = [self.x[k].x for k in self.x]
        # objective value
        obj = self._model.objVal
        return sol, obj

    def copy(self):
        """
        A method to copy model

        Returns:
            optModel: new copied model
        """
        new_model = copy(self)
        # update model
        self._model.update()
        # new model
        new_model._model = self._model.copy()
        # variables for new model
        x = new_model._model.getVars()
        new_model.x = {key: x[i] for i, key in enumerate(x)}
        return new_model

    def addConstr(self, coefs, rhs):
        """
        A method to add new constraint

        Args:
            coefs (np.ndarray / list): coeffcients of new constraint
            rhs (float): right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        if len(coefs) != self.num_cost:
            raise ValueError("Size of coef vector cannot cost.")
        # copy
        new_model = self.copy()
        # add constraint
        expr = gp.quicksum(coefs[i] * new_model.x[k]
                           for i, k in enumerate(new_model.x)) <= rhs
        new_model._model.addConstr(expr)
        return new_model
