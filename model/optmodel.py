#!/usr/bin/env python
# coding: utf-8

from abc import ABC, abstractmethod

class optModel(ABC):
    """abstract class for optimization model"""

    def __init__(self):
        self.model = self._getModel()

    @abstractmethod
    def _getModel(self):
        """
        build a model from a optimization solver
        """
        pass

    @abstractmethod
    def setObj(self, c):
        """
        set objective function with given cost vector
        """
        pass

    @abstractmethod
    def solve(self):
        """
        solve model
        Returns:
            sol: optimal solution
        """
        pass
