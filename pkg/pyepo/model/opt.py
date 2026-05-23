#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch

from pyepo import EPO


def costToNumpy(c: np.ndarray | torch.Tensor | list, dtype=np.float32) -> np.ndarray:
    """
    Normalize a cost vector to a numpy array, detaching torch tensors as needed.

    Args:
        c (np.ndarray / list / torch.Tensor): cost vector
        dtype (np.dtype): target dtype when ``c`` is not already a tensor; torch
            tensors are converted via ``.detach().cpu().numpy()`` and keep their
            existing dtype.

    Returns:
        np.ndarray: numpy cost vector
    """
    if isinstance(c, torch.Tensor):
        return c.detach().cpu().numpy()
    return np.asarray(c, dtype=dtype)

class optModel(ABC):
    """
    This is an abstract class for an optimization model

    Attributes:
        _model (optimization model): underlying solver model object
    """

    def __init__(self) -> None:
        # default sense
        if not hasattr(self, "modelSense"):
            self.modelSense = EPO.MINIMIZE
        self._model, self.x = self._getModel()

    def __repr__(self) -> str:
        return 'optModel ' + self.__class__.__name__

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
            c (ndarray): cost of objective function
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
            coefs (ndarray): coefficients of the new constraint
            rhs (float): right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        raise NotImplementedError

    def relax(self) -> optModel:
        """
        A method to relax the MIP model. Subclasses should override.
        """
        raise RuntimeError("Method 'relax' is not implemented.")


class unionFind:
    """
    Union-Find data structure for cycle detection in graphs
    """
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i
            return True
        return False


def getTspTour(
    edge_list: list[tuple[int, int]],
    num_nodes: int,
    sol: np.ndarray | torch.Tensor | list,
    threshold: float = 1e-2,
) -> list[int]:
    """
    Reconstruct a TSP tour from an undirected edge-selection vector.

    Args:
        edge_list (list): undirected edges of the model, ordered to match ``sol``
        num_nodes (int): number of nodes in the TSP instance
        sol (sequence): solution values aligned with ``edge_list``; an edge is
            considered active when its value exceeds ``threshold``
        threshold (float): activation threshold for an edge

    Returns:
        list: node sequence of the tour (closes back to node 0 if reachable)

    Raises:
        ValueError: if the solution does not form a single connected tour
            (skips nodes or contains disconnected subtours).
    """
    # active edges
    edges = defaultdict(list)
    for i, (j, k) in enumerate(edge_list):
        if sol[i] > threshold:
            edges[j].append(k)
            edges[k].append(j)
    # all nodes must appear in the active edge set
    if len(edges) != num_nodes:
        raise ValueError(
            f"Solution does not cover all {num_nodes} nodes (got {len(edges)}); "
            "the model returned an infeasible TSP solution.")
    # walk the tour starting from the first node with an active edge
    start = list(edges.keys())[0]
    visited = {start}
    tour = [start]
    while len(visited) < len(edges):
        i = tour[-1]
        for j in edges[i]:
            if j not in visited:
                tour.append(j)
                visited.add(j)
                break
        else:
            # no unvisited neighbour: solution contains a subtour
            raise ValueError(
                "Solution contains disconnected subtours; cannot form a single "
                "tour. Check that subtour elimination constraints are active.")
    if 0 in edges[tour[-1]]:
        tour.append(0)
    return tour


def _get_grid_arcs(grid: tuple[int, int]) -> list[tuple[int, int]]:
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
