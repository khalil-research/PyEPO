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
    from typing_extensions import Self

    from pyepo.EPO import ModelSense


class optModel(ABC):
    """
    Abstract base class for predict-then-optimize models.

    Subclasses wrap an optimization solver or algorithm with a unified
    ``_getModel`` / ``setObj`` / ``solve`` / ``num_cost`` interface that
    ``pyepo.func`` modules call during training. Concrete backends are
    provided for GurobiPy (``optGrbModel``), Pyomo (``optOmoModel``),
    COPT (``optCoptModel``), OR-Tools (``optOrtModel`` / ``optOrtCpModel``),
    and MPAX (``optMpaxModel``); subclass ``optModel`` directly to integrate
    any other solver or algorithm.

    The default objective sense is minimization; set
    ``self.modelSense = EPO.MAXIMIZE`` in ``_getModel`` or ``__init__`` for
    maximization problems (some backends, e.g. Gurobi and COPT, detect this
    automatically from the underlying solver model).

    Attributes:
        _model (optimization model): underlying solver model object
        modelSense (ModelSense): EPO.MINIMIZE (default) or EPO.MAXIMIZE
    """

    modelSense: ModelSense = EPO.MINIMIZE
    # populated by problem-level bases (shortestPathBase / tspABBase) or _getModel
    arcs: list = []
    _cost_vars: list = []

    def __init__(self) -> None:
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

    @property
    def c_pred_index(self) -> np.ndarray | None:
        """Variable positions the predicted cost lands on, or ``None`` when every variable is predicted (the default)."""
        return None

    def _fullCost(self, pred_cost):
        """The full objective coefficient for a predicted cost; identity by default, overridden under partial prediction."""
        return pred_cost

    def copy(self) -> Self:
        """
        A method to copy the model

        Returns:
            optModel: new copied model
        """
        new_model = deepcopy(self)
        return new_model

    def addConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> Self:
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
