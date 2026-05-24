#!/usr/bin/env python
"""
Abstract optimization model
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING

from pyepo import EPO

if TYPE_CHECKING:
    import numpy as np
    import torch

    from pyepo.EPO import ModelSense


class optModel(ABC):
    """
    This is an abstract class for an optimization model

    Attributes:
        _model (optimization model): underlying solver model object
    """

    modelSense: ModelSense

    def __init__(self) -> None:
        # default sense
        if not hasattr(self, "modelSense"):
            self.modelSense = EPO.MINIMIZE
        self._model, self.x = self._getModel()

    def __repr__(self) -> str:
        return "optModel " + self.__class__.__name__

    @property
    def num_cost(self) -> int:
        """
        number of costs to be predicted
        """
        return len(self.x)

    @abstractmethod
    def _getModel(self) -> tuple:
        """
        An abstract method to build a model from an optimization solver

        Returns:
            tuple: optimization model and variables
        """
        raise NotImplementedError

    @abstractmethod
    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        """
        An abstract method to set the objective function

        Args:
            c: cost of objective function
        """
        raise NotImplementedError

    @abstractmethod
    def solve(self) -> tuple[np.ndarray | torch.Tensor | list, float]:
        """
        An abstract method to solve the model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        raise NotImplementedError

    def copy(self) -> optModel:
        """
        A method to copy the model

        Returns:
            optModel: new copied model
        """
        new_model = deepcopy(self)
        return new_model

    def addConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> optModel:
        """
        A method to add a new constraint. Subclasses should override.

        Args:
            coefs: coefficients of the new constraint
            rhs: right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        raise NotImplementedError

    def relax(self) -> optModel:
        """
        A method to relax the MIP model. Subclasses should override.
        """
        raise NotImplementedError("Method 'relax' is not implemented.")
