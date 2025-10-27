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

    def _objective_fun(self, c):
        """
        Default objective function f(x, c) = cᵀx.
        Can be overridden in subclasses for nonlinear objectives.

        Args:
            c (np.ndarray): cost vector for one sample

        Returns:
            Gurobi linear expression representing f(z, c)
        """
        if isinstance(self.x, gp.MVar):
            return c @ self.x
        return gp.quicksum(c[j] * self.x[j] for j in range(len(c)))

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
        self._model.setObjective(self._objective_fun(c))

    def setWeightObj(self, w, c):
        """
        Set a weighted objective for predictive prescriptions.

        Args:
            w (np.ndarray): shape (N,), weights for each sample
            c (np.ndarray): shape (N, C), cost vectors for each sample
        """
        if c.shape[1] != self.num_cost:
            raise ValueError("Cost vector dimension mismatch.")
        if c.shape[0] != len(w):
            raise ValueError("Weights and costs must have same first dimension.")
        
        # Check if PyTorch tensor inputs
        if isinstance(w, torch.Tensor):
            w = w.detach().cpu().numpy()
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()

        obj = gp.quicksum(w[i] * self._objective_fun(c[i]) for i in range(len(w)))

        self._model.setObjective(obj)

    def cal_obj(self, c, x):
        """"
        An abstract method to calculate the objective value

        Args:
            c (ndarray): cost of objective
            x (ndarray): the decision variables
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector cannot match vars.")

        # check if c is a PyTorch tensor
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        else:
            c = np.asarray(c, dtype=np.float32)

        # check if c is a PyTorch tensor
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x, dtype=np.float32)

        obj = c @ x

        return obj

            

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
