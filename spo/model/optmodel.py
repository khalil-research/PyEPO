#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model
"""

from abc import ABC, abstractmethod
from copy import copy


class optModel(ABC):
    """
    This is an abstract class for optimization model
    """

    def __init__(self):
        self._model = self._getModel()

    def __repr__(self):
        return self.__class__.__name__

    @property
    @abstractmethod
    def num_cost(self):
        """
        number of cost to be predicted
        """
        raise NotImplementedError

    @abstractmethod
    def _getModel(self):
        """
        An abstract method to build a model from a optimization solver
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
        new_model = copy(self)
        return new_model

    @abstractmethod
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
