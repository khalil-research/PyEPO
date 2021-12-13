#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on gurobipy
"""

from copy import copy
import gurobipy as gp
from gurobipy import GRB

from spo.model import optModel


class optGRBModel(optModel):
    """
    This is an abstract class for Gurobi-based optimization model
    """

    def __init__(self):
        super().__init__()
        # turn off output
        self._model.Params.outputFlag = 0

    def __repr__(self):
        return "optGRBModel " + self.__class__.__name__

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (ndarray): cost of objective function
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector cannot match vars.")
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
        return [self.x[k].x for k in self.x], self._model.objVal

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
        new_model.x = {key: x[i] for i, key in enumerate(self.x)}
        return new_model

    def addConstr(self, coefs, rhs):
        """
        A method to add new constraint

        Args:
            coefs (ndarray): coeffcients of new constraint
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
