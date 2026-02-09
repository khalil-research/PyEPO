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
    This is an abstract class for an optimization model

    Attributes:
        _model (optimization model): underlying solver model object
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
        number of costs to be predicted
        """
        return len(self.x)

    @abstractmethod
    def _getModel(self):
        """
        An abstract method to build a model from an optimization solver

        Returns:
            tuple: optimization model and variables
        """
        raise NotImplementedError

    @abstractmethod
    def setObj(self, c):
        """
        An abstract method to set the objective function

        Args:
            c (ndarray): cost of objective function
        """
        raise NotImplementedError

    @abstractmethod
    def solve(self):
        """
        An abstract method to solve the model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        raise NotImplementedError

    def copy(self):
        """
        A method to copy the model

        Returns:
            optModel: new copied model
        """
        new_model = deepcopy(self)
        return new_model

    def addConstr(self, coefs, rhs):
        """
        An abstract method to add a new constraint

        Args:
            coefs (ndarray): coefficients of the new constraint
            rhs (float): right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        raise NotImplementedError

    def relax(self):
        """
        An unimplemented method to relax the MIP model
        """
        raise RuntimeError("Method 'relax' is not implemented.")


def _get_grid_arcs(grid):
    """
    Get list of arcs for a grid network.

    Args:
        grid (tuple of int): size of grid network (rows, cols)

    Returns:
        list: arcs as (source, target) tuples
    """
    arcs = []
    for i in range(grid[0]):
        for j in range(grid[1] - 1):
            v = i * grid[1] + j
            arcs.append((v, v + 1))
        if i == grid[0] - 1:
            continue
        for j in range(grid[1]):
            v = i * grid[1] + j
            arcs.append((v, v + grid[1]))
    return arcs
