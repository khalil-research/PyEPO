#!/usr/bin/env python
"""
Abstract optimization model
"""

from __future__ import annotations

import functools
import inspect
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pyepo import EPO

if TYPE_CHECKING:
    import numpy as np
    import torch
    from typing_extensions import Self

    from pyepo.EPO import ModelSense


@dataclass(frozen=True, init=False)
class ModelSpec:
    """Serializable recipe for building a fresh optimization model."""

    model_type: type[optModel]
    _config: dict

    def __init__(self, model_type: type[optModel], config: dict) -> None:
        object.__setattr__(self, "model_type", model_type)
        object.__setattr__(self, "_config", deepcopy(config))

    @property
    def config(self) -> dict:
        """Return an independent copy of the constructor configuration."""
        return deepcopy(self._config)

    def build(self) -> optModel:
        """Build a fresh model without sharing mutable configuration values."""
        return self.model_type.from_config(self._config)


def _capture_init_config(init, args, kwargs) -> dict:
    """Flatten a constructor call into keyword arguments that rebuild the model."""
    sig = inspect.signature(init)
    bound = sig.bind(None, *args, **kwargs)
    config = {}
    for i, (name, value) in enumerate(bound.arguments.items()):
        # the first bound argument is self
        if i == 0:
            continue
        kind = sig.parameters[name].kind
        # **kwargs: merge captured keywords in directly
        if kind is inspect.Parameter.VAR_KEYWORD:
            config.update(deepcopy(value))
        # nameless positionals cannot replay by keyword; override get_config to keep them
        elif kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.POSITIONAL_ONLY):
            continue
        else:
            config[name] = deepcopy(value)
    return config


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
    arcs: list
    _cost_vars: list

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # only wrap a subclass that defines its own __init__
        if "__init__" not in cls.__dict__:
            return
        user_init = cls.__init__

        @functools.wraps(user_init)
        def _init_capturing(self, *args, **kwargs):
            # record only the outermost call; nested super().__init__ leaves it intact
            if "_init_config" not in self.__dict__:
                self._init_config = _capture_init_config(user_init, args, kwargs)
            user_init(self, *args, **kwargs)

        cls.__init__ = _init_capturing

    def __init__(self) -> None:
        # Cache for models whose solver variables do not map one-to-one to
        # predicted costs (for example directed TSP/VRP formulations).
        # It must be instance-local because concrete builders mutate it.
        self._cost_vars: list = []
        # user cuts tracked for replay on relax
        self._extra_constrs: list = []
        self._model, self.x = self._getModel()

    def __repr__(self) -> str:
        return "optModel " + self.__class__.__name__

    def get_config(self) -> dict:
        """Return the constructor configuration for this model."""
        return deepcopy(self.__dict__.get("_init_config", {}))

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """Build a model from a configuration produced by ``get_config``."""
        return cls(**deepcopy(config))

    def to_spec(self) -> ModelSpec:
        """Return a serializable, immutable-snapshot rebuild recipe."""
        return ModelSpec(type(self), self.get_config())

    def rebuild(self) -> Self:
        """Build a structurally equivalent model with clean runtime state."""
        return cast("Self", self.to_spec().build())

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

    def _setFullObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        """Set the objective from full-space coefficients; alias of ``setObj`` unless overridden under partial prediction."""
        self.setObj(c)

    def copy(self) -> Self:
        """
        A method to copy the model

        Returns:
            optModel: new copied model
        """
        return deepcopy(self)

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
