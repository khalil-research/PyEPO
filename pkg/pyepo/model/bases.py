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

import math
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np

from pyepo import EPO
from pyepo.model._common import validate_constraint
from pyepo.model.opt import optModel
from pyepo.model.utils import _EDGE_ACTIVE_TOL, _get_grid_arcs, getTspTour
from pyepo.utils import costToNumpy

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


def _real_array(value, name: str, *, ndim: int) -> np.ndarray:
    """Return a finite real array with the requested dimensionality."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a NumPy-compatible array.") from exc
    if array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional array.")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain real numbers.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return np.array(array, copy=True)


def _positive_int(value, name: str, *, minimum: int = 1) -> int:
    """Validate a non-boolean integer lower bound."""
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    return int(value)


def _finite_real(value, name: str, *, positive: bool = False) -> float:
    """Validate a finite non-boolean real scalar."""
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number.")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real number.")
    if positive and scalar <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return scalar


class shortestPathBase(optModel):
    """
    Problem-level base for grid shortest path.

    Finds the minimum-cost path from the northwest corner to the southeast
    corner of an ``(h, w)`` grid network. The problem is formulated as a
    minimum-cost flow LP with arc-incidence equality constraints; arcs are
    enumerated automatically from the grid dimensions and stored on
    ``self.arcs``.

    Attributes:
        grid (tuple of int): grid dimensions (rows, cols)
        arcs (list): ordered list of (source, target) arcs in the grid
    """

    def __init__(self, grid: tuple[int, int], *args, **kwargs) -> None:
        """
        Args:
            grid: grid dimensions ``(rows, cols)``
        """
        try:
            dimensions = tuple(grid)
        except TypeError as exc:
            raise ValueError("grid must contain exactly two positive integer dimensions.") from exc
        if (
            len(dimensions) != 2
            or any(
                not isinstance(value, Integral) or isinstance(value, bool) or value < 1
                for value in dimensions
            )
            or dimensions[0] * dimensions[1] < 2
        ):
            raise ValueError("grid must contain exactly two positive integer dimensions.")
        self.grid = (int(dimensions[0]), int(dimensions[1]))
        self.arcs = _get_grid_arcs(self.grid)
        super().__init__(*args, **kwargs)

    def get_config(self) -> dict:
        return {**super().get_config(), "grid": self.grid}

    @property
    def num_cost(self) -> int:
        return len(self.arcs)


class knapsackBase(optModel):
    """
    Problem-level base for the multi-dimensional knapsack.

    Selects a subset of items that maximizes total value subject to
    per-resource capacity constraints. Items, dimensions, and capacities are
    inferred from the shapes of ``weights`` and ``capacity``: ``weights``
    has shape ``(dim, n_items)`` and ``capacity`` has length ``dim``. Item
    values are the predicted cost coefficients ``c`` set via ``setObj``.

    Attributes:
        weights (np.ndarray): item weights, shape ``(dim, n_items)``
        capacity (np.ndarray): per-dimension capacity, shape ``(dim,)``
        items (list): item indices ``0 .. n_items - 1``
    """

    modelSense = EPO.MAXIMIZE

    def __init__(
        self,
        weights: np.ndarray | list,
        capacity: np.ndarray | list,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            weights: item weights with shape ``(dim, n_items)``
            capacity: per-dimension capacity with length ``dim``
        """
        self.weights = _real_array(weights, "weights", ndim=2)
        self.capacity = _real_array(capacity, "capacity", ndim=1)
        if self.weights.shape[0] == 0 or self.weights.shape[1] == 0:
            raise ValueError("weights must contain at least one dimension and one item.")
        if len(self.capacity) != self.weights.shape[0]:
            raise ValueError("capacity length must match the number of weight dimensions.")
        self.items = list(range(self.weights.shape[1]))
        super().__init__(*args, **kwargs)

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "weights": self.weights.copy(),
            "capacity": self.capacity.copy(),
        }

    @property
    def num_cost(self) -> int:
        return len(self.items)


class portfolioBase(optModel):
    """
    Problem-level base for Markowitz mean-variance portfolio optimization.

    Allocates a unit budget across ``num_assets`` to maximize predicted
    expected return subject to a quadratic risk budget
    :math:`\\mathbf{x}^\\top \\boldsymbol{\\Sigma} \\mathbf{x} \\le \\gamma\\,
    \\overline{\\boldsymbol{\\Sigma}}`, where :math:`\\overline{\\boldsymbol{\\Sigma}}`
    is the mean covariance entry. Predicted asset returns are the cost
    coefficients ``c`` set via ``setObj``; the covariance and risk budget
    are fixed across instances.

    Attributes:
        num_assets (int): number of assets
        covariance (np.ndarray): asset-return covariance matrix
        gamma (float): risk tolerance multiplier
        risk_level (float): derived risk budget = ``gamma * mean(covariance)``
    """

    modelSense = EPO.MAXIMIZE

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
            covariance: covariance matrix of the asset returns
            gamma: risk tolerance multiplier on the mean covariance
        """
        self.num_assets = _positive_int(num_assets, "num_assets")
        self.covariance = _real_array(covariance, "covariance", ndim=2)
        if self.covariance.shape != (self.num_assets, self.num_assets):
            raise ValueError("covariance shape must be (num_assets, num_assets).")
        self.gamma = _finite_real(gamma, "gamma")
        if self.gamma < 0:
            raise ValueError("gamma must be greater than or equal to zero.")
        super().__init__(*args, **kwargs)

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "num_assets": self.num_assets,
            "covariance": self.covariance.copy(),
            "gamma": self.gamma,
        }

    @property
    def num_cost(self) -> int:
        return self.num_assets

    @property
    def risk_level(self) -> float:
        # risk budget the backends plug into x^T Σ x <= risk_level
        return self.gamma * np.mean(self.covariance)


class tspABBase(optModel):
    """
    Problem-level base for the symmetric traveling-salesperson problem.

    Finds the shortest tour visiting each of ``num_nodes`` cities exactly
    once and returning to the origin. Three concrete ILP formulations are
    supplied per backend:

    * **DFJ** (Dantzig-Fulkerson-Johnson) -- lazy subtour-elimination
      constraints via solver callbacks (no LP relaxation).
    * **GG** (Gavish-Graves) -- flow-based formulation with auxiliary
      flow variables.
    * **MTZ** (Miller-Tucker-Zemlin) -- compact formulation with per-node
      potential auxiliaries.

    The base only manages formulation-independent state (nodes, edges,
    extra-constraint replay) and the ``getTour`` helper. Concrete
    formulations implement ``_getModel`` (build the solver model) and
    ``_addExtraConstr`` (add a single linear extra constraint).

    Attributes:
        num_nodes (int): number of nodes
        nodes (list): node indices ``0 .. num_nodes - 1``
        edges (list): undirected edges as ``(i, j)`` with ``i < j``
    """

    def __init__(self, num_nodes: int, *args, **kwargs) -> None:
        """
        Args:
            num_nodes: number of nodes (cities)
        """
        self.num_nodes = _positive_int(num_nodes, "num_nodes", minimum=3)
        self.nodes = list(range(self.num_nodes))
        self.edges = list(combinations(self.nodes, 2))
        self._extra_constrs: list = []
        super().__init__(*args, **kwargs)

    def get_config(self) -> dict:
        return {**super().get_config(), "num_nodes": self.num_nodes}

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

    def copy(self) -> Self:
        """
        Return a fresh model with all extra constraints replayed onto it.
        """
        new_model = self.rebuild()
        self._replay_extras(new_model)
        copy_objective = getattr(self, "_copy_objective_to", None)
        if copy_objective is not None:
            copy_objective(new_model)
        return new_model

    def _replay_extras(self, other: tspABBase) -> None:
        for coefs, rhs in self._extra_constrs:
            snapshot = coefs.copy()
            other._extra_constrs.append((snapshot, rhs))
            other._addExtraConstr(snapshot, rhs)

    def addConstr(
        self,
        coefs: np.ndarray | torch.Tensor | list,
        rhs: float,
    ) -> Self:
        """
        Return a new model with one extra linear constraint added.

        Args:
            coefs: per-edge coefficients aligned with ``self.edges``
            rhs: right-hand side
        """
        rhs = validate_constraint(coefs, rhs, self.num_cost)
        # normalize to numpy so replay on copy/relax avoids per-element GPU sync
        coefs = costToNumpy(coefs).copy()
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


class vrpABBase(optModel):
    """
    Problem-level base for the capacitated vehicle routing problem.

    Routes a fleet of ``num_vehicle`` vehicles of capacity ``capacity`` from
    a depot (node 0) to serve every customer's ``demands`` exactly once at
    minimum total edge cost. Concrete formulations (RCI with lazy cuts, MTZ
    with load potentials) are supplied per backend.

    The base manages formulation-independent state (nodes, edges, demands,
    extra-constraint replay) and the ``getTour`` helper. Concrete
    formulations implement ``_getModel`` (build the solver model) and
    ``_addExtraConstr`` (add a single linear extra constraint over the
    cost-aligned edge variables).

    Attributes:
        num_nodes (int): number of nodes (depot at index 0)
        nodes (list): node indices ``0 .. num_nodes - 1``
        edges (list): undirected edges as ``(i, j)`` with ``i < j``
        demands (list | np.ndarray): per-customer demands, length ``num_nodes - 1``
        capacity (float): per-vehicle capacity
        num_vehicle (int): number of vehicles
    """

    def __init__(
        self,
        num_nodes: int,
        demands: list[float] | np.ndarray,
        capacity: float,
        num_vehicle: int,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            num_nodes: number of nodes (depot is node 0)
            demands: per-customer demands, length ``num_nodes - 1``
            capacity: vehicle capacity
            num_vehicle: number of vehicles
        """
        # problem parameters
        self.num_nodes = _positive_int(num_nodes, "num_nodes", minimum=2)
        self.nodes = list(range(self.num_nodes))
        self.edges = [(i, j) for i in self.nodes for j in self.nodes if i < j]
        demand_array = _real_array(demands, "demands", ndim=1)
        if len(demand_array) != self.num_nodes - 1:
            raise ValueError("demands length must equal num_nodes - 1.")
        if (demand_array < 0).any():
            raise ValueError("demands must be nonnegative.")
        self.demands = demand_array.tolist() if isinstance(demands, list) else demand_array
        self.capacity = _finite_real(capacity, "capacity", positive=True)
        self.num_vehicle = _positive_int(num_vehicle, "num_vehicle")
        self._extra_constrs: list = []
        super().__init__(*args, **kwargs)

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "num_nodes": self.num_nodes,
            "demands": deepcopy(self.demands),
            "capacity": self.capacity,
            "num_vehicle": self.num_vehicle,
        }

    @property
    def num_cost(self) -> int:
        # one predicted cost per undirected edge
        return len(self.edges)

    def getTour(self, sol: np.ndarray | torch.Tensor | list) -> list[list[int]]:
        """
        Reconstruct vehicle tours from an undirected edge-selection vector.

        Args:
            sol: per-edge selection values aligned with ``self.edges``

        Returns:
            list of tours; each tour is a node sequence starting and ending at the depot
        """
        # active-edge adjacency
        adj: dict[int, list[int]] = defaultdict(list)
        for i, (u, v) in enumerate(self.edges):
            if sol[i] > _EDGE_ACTIVE_TOL:
                adj[u].append(v)
                adj[v].append(u)
        # peel one depot-anchored route at a time
        routes = []
        while adj[0]:
            v_curr = 0
            tour = [0]
            v_next = adj[v_curr][0]
            adj[v_curr].remove(v_next)
            adj[v_next].remove(v_curr)
            while v_next != 0:
                tour.append(v_next)
                # single-customer dead-end falls back to depot
                if not adj[v_next]:
                    v_curr, v_next = v_next, 0
                else:
                    v_curr, v_next = v_next, adj[v_next][0]
                    adj[v_curr].remove(v_next)
                    adj[v_next].remove(v_curr)
            tour.append(0)
            routes.append(tour)
        return routes

    def copy(self) -> Self:
        """
        Return a fresh model with all extra constraints replayed onto it.
        """
        new_model = self.rebuild()
        self._replay_extras(new_model)
        copy_objective = getattr(self, "_copy_objective_to", None)
        if copy_objective is not None:
            copy_objective(new_model)
        return new_model

    def _replay_extras(self, other: vrpABBase) -> None:
        # re-add tracked extra constraints to a fresh copy
        for coefs, rhs in self._extra_constrs:
            snapshot = coefs.copy()
            other._extra_constrs.append((snapshot, rhs))
            other._addExtraConstr(snapshot, rhs)

    def addConstr(
        self,
        coefs: np.ndarray | torch.Tensor | list,
        rhs: float,
    ) -> Self:
        """
        Return a new model with one extra linear constraint added.

        Args:
            coefs: per-edge coefficients aligned with ``self.edges``
            rhs: right-hand side
        """
        rhs = validate_constraint(coefs, rhs, self.num_cost)
        # normalize to numpy so replay on copy avoids per-element solver-var sync
        coefs = costToNumpy(coefs).copy()
        new_model = self.copy()
        new_model._extra_constrs.append((coefs, rhs))
        new_model._addExtraConstr(coefs, rhs)
        return new_model

    def _expand_coefs(self, coefs: np.ndarray) -> np.ndarray:
        # per-cost-var coefficients; override for paired (directed-edge) formulations
        return coefs

    def _addExtraConstr(
        self,
        coefs: np.ndarray | torch.Tensor | list,
        rhs: float,
    ) -> None:
        """Backend-specific: add a single linear constraint over the cost-aligned edge variables."""
        raise NotImplementedError
