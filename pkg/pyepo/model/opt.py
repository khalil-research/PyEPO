#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model
"""

from abc import ABC, abstractmethod
from copy import deepcopy

from pyepo import EPO

class optModel(ABC):
    """
    This is an abstract class for optimization model

    Attributes:
        _model (GurobiPy model): Gurobi model
    """

    def __init__(self):
        # default sense
        if not hasattr(self, "modelSense"):
            self.modelSense = EPO.MINIMIZE
        self._model, self.x = self._getModel()

    def __repr__(self):
        return 'optModel ' + self.__class__.__name__

    @property
    def num_cost(self):
        """
        number of cost to be predicted
        """
        return len(self.x)

    @abstractmethod
    def _getModel(self):
        """
        An abstract method to build a model from a optimization solver

        Returns:
            tuple: optimization model and variables
        """
        raise NotImplementedError

    @abstractmethod
    def setObj(self, c):
        """
        An abstract method to set objective function

        Args:
            c (ndarray): cost of objective function
        """
        raise NotImplementedError
    
    @abstractmethod
    def setWeightObj(self, w, c):
        """
        Set a weighted objective for predictive prescriptions.

        Args:
            w (np.ndarray): shape (N,), weights for each sample
            c (np.ndarray): shape (N, C), cost vectors for each sample
        """
        raise NotImplementedError
    
    @abstractmethod
    def cal_obj(self, c, x):
        """"
        An abstract method to calculate the objective value

        Args:
            c (ndarray): cost of objective
            x (ndarray): the decision variables
        """
        raise NotImplementedError

    @abstractmethod
    def solve(self):
        """
        An abstract method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        raise NotImplementedError

    def copy(self):
        """
        An abstract method to copy model

        Returns:
            optModel: new copied model
        """
        new_model = deepcopy(self)
        return new_model

    def addConstr(self, coefs, rhs):
        """
        An abstract method to add new constraint

        Args:
            coefs (ndarray): coeffcients of new constraint
            rhs (float): right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        raise NotImplementedError

    def relax(self):
        """
        A unimplemented method to relax MIP model
        """
        raise RuntimeError("Method 'relax' is not implemented.")
