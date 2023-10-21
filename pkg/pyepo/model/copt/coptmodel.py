#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on Cardinal Optimizer（COPT）
"""

from copy import copy

from coptpy import COPT

from pyepo import EPO
from pyepo.model.opt import optModel


class optCoptModel(optModel):
    """
    This is an abstract class for Cardinal Optimizer optimization model

    Attributes:
        _model (COPT model): COPT model
    """

    def __init__(self):
        """
        Args:
            solver (str): optimization solver in the background
        """
        super().__init__()
        # model sense
        if self._model.ObjSense == COPT.MINIMIZE:
            self.modelSense = EPO.MINIMIZE
        if self._model.ObjSense == COPT.MAXIMIZE:
            self.modelSense = EPO.MAXIMIZE
        # turn off output
        self._model.setParam("Logging", 0)

    def __repr__(self):
        return "optCoptModel " + self.__class__.__name__

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (np.ndarray / list): cost of objective function
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector cannot match vars.")
        # set obj
        obj = sum(c[i] * self.x[k] for i, k in enumerate(self.x))
        self._model.setObjective(obj)

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        # solve
        self._model.solve()
        return [var.x for var in self._model.getVars()], self._model.objVal

    def copy(self):
        """
        A method to copy model

        Returns:
            optModel: new copied model
        """
        new_model = copy(self)
        # new model
        new_model._model = self._model.clone()
        # variables for new model
        x = new_model._model.getVars()
        new_model.x = {key: x[i] for i, key in enumerate(self.x)}
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
        expr = sum(coefs[i] * new_model.x[k]
                   for i, k in enumerate(new_model.x)) <= rhs
        new_model._model.addConstr(expr)
        return new_model
