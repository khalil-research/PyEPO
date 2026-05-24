#!/usr/bin/env python
"""
Problem-level base classes for optimization models.

Each base captures the problem-specific state (data, indices, derived attrs)
and methods that are independent of the underlying solver. Concrete classes
combine a problem base with a backend base via multiple inheritance, with the
problem base listed first so its ``__init__`` runs before the backend's::

    class shortestPathModel(shortestPathBase, optGrbModel):
        def _getModel(self):
            ...  # solver-specific build
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from pyepo import EPO
from pyepo.model.opt import optModel
from pyepo.model.utils import _get_grid_arcs, getTspTour

if TYPE_CHECKING:
    import torch


class shortestPathBase(optModel):
    """
    Problem-level base for grid shortest path.

    Attributes:
        grid (tuple of int): grid dimensions (rows, cols)
        arcs (list): list of (source, target) arcs in the grid
    """

    def __init__(self, grid: tuple[int, int], *args, **kwargs) -> None:
        """
        Args:
            grid: grid dimensions (rows, cols)
        """
        self.grid = grid
        self.arcs = _get_grid_arcs(grid)
        super().__init__(*args, **kwargs)

    @property
    def num_cost(self) -> int:
        return len(self.arcs)


class knapsackBase(optModel):
    """
    Problem-level base for multi-dim knapsack.

    Attributes:
        weights (np.ndarray): item weights, shape (dim, n_items)
        capacity (np.ndarray): per-dimension capacity
        items (list): item indices
    """

    def __init__(
        self,
        weights: np.ndarray | list,
        capacity: np.ndarray | list,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            weights: weights of items
            capacity: total capacity
        """
        self.weights = np.asarray(weights)
        self.capacity = np.asarray(capacity)
        self.items = list(range(self.weights.shape[1]))
        self.modelSense = EPO.MAXIMIZE
        super().__init__(*args, **kwargs)

    @property
    def num_cost(self) -> int:
        return len(self.items)


class portfolioBase(optModel):
    """
    Problem-level base for Markowitz portfolio.

    Attributes:
        num_assets (int): number of assets
        covariance (np.ndarray): asset return covariance matrix
        risk_level (float): risk budget = gamma * mean(covariance)
    """

    def __init__(
        self,
        num_assets: int,
        covariance: np.ndarray,
        gamma: float = 2.25,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            num_assets: number of assets
            covariance: covariance matrix of the returns
            gamma: risk level parameter
        """
        self.num_assets = num_assets
        self.covariance = np.asarray(covariance)
        self.risk_level = gamma * np.mean(self.covariance)
        self.modelSense = EPO.MAXIMIZE
        super().__init__(*args, **kwargs)

    @property
    def num_cost(self) -> int:
        return self.num_assets


class tspABBase(optModel):
    """
    Problem-level base for TSP (formulation-independent state and helpers).

    Concrete formulations (GG, MTZ, DFJ) only need to implement
    ``_getModel`` (build the solver model) and ``_addExtraConstr``
    (add a single linear extra constraint to ``self._model``).

    Attributes:
        num_nodes (int): number of nodes
        nodes (list): node indices 0..num_nodes-1
        edges (list): undirected edges as (i, j) with i < j
    """

    def __init__(self, num_nodes: int, *args, **kwargs) -> None:
        """
        Args:
            num_nodes: number of nodes
        """
        self.num_nodes = num_nodes
        self.nodes = list(range(num_nodes))
        self.edges = list(combinations(self.nodes, 2))
        self._extra_constrs: list = []
        super().__init__(*args, **kwargs)

    @property
    def num_cost(self) -> int:
        # use edges; backend's self.x has 2*num_edges directed Vars
        return len(self.edges)

    def getTour(self, sol: np.ndarray | torch.Tensor | list) -> list[int]:
        """
        Reconstruct a TSP tour from an undirected edge-selection vector.

        Raises:
            ValueError: if the solution does not form a single connected tour.
        """
        return getTspTour(self.edges, self.num_nodes, sol)

    def copy(self) -> tspABBase:
        """
        Return a fresh model with all extra constraints replayed onto it.
        """
        new_model = self._new_instance()
        self._replay_extras(new_model)
        return new_model

    def _new_instance(self) -> tspABBase:
        """Construct a fresh instance with the same problem args. Override for backends with extra ctor args."""
        return type(self)(self.num_nodes)

    def _replay_extras(self, other: tspABBase) -> None:
        for coefs, rhs in self._extra_constrs:
            other._extra_constrs.append((coefs, rhs))
            other._addExtraConstr(coefs, rhs)

    def addConstr(
        self,
        coefs: np.ndarray | torch.Tensor | list,
        rhs: float,
    ) -> tspABBase:
        """
        Return a new model with one extra linear constraint added.

        Args:
            coefs: per-edge coefficients aligned with ``self.edges``
            rhs: right-hand side
        """
        if len(coefs) != self.num_cost:
            raise ValueError("Size of coef vector does not match number of cost variables.")
        # normalize to numpy so replay on copy/relax avoids per-element GPU sync
        coefs = np.asarray(coefs)
        new_model = self.copy()
        new_model._extra_constrs.append((coefs, rhs))
        new_model._addExtraConstr(coefs, rhs)
        return new_model

    def _addExtraConstr(
        self,
        coefs: np.ndarray | torch.Tensor | list,
        rhs: float,
    ) -> None:
        """Backend-specific: add a single linear constraint to ``self._model``."""
        raise NotImplementedError
