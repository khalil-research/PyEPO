#!/usr/bin/env python
"""
Problem-level helpers for optimization model construction.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


# edge/arc treated as inactive when solver value falls below this (≈ MIP feasibility tol)
_EDGE_ACTIVE_TOL: float = 1e-6


def _get_grid_arcs(grid: tuple[int, int]) -> list[tuple[int, int]]:
    """
    Get list of arcs for a grid network.

    Args:
        grid: size of grid network (rows, cols)

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
    threshold: float = _EDGE_ACTIVE_TOL,
) -> list[int]:
    """
    Reconstruct a TSP tour from an undirected edge-selection vector.

    Args:
        edge_list: undirected edges of the model, ordered to match ``sol``
        num_nodes: number of nodes in the TSP instance
        sol: solution values aligned with ``edge_list``; an edge is
            considered active when its value exceeds ``threshold``
        threshold: activation threshold for an edge

    Returns:
        list: node sequence of the tour, starting and ending at node 0

    Raises:
        ValueError: if the solution does not form a single Hamiltonian tour
            closed back to node 0 (skips nodes, contains disconnected subtours,
            or the last visited node is not adjacent to 0).
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
            "the model returned an infeasible TSP solution."
        )
    visited = {0}
    tour = [0]
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
                "tour. Check that subtour elimination constraints are active."
            )
    if 0 not in edges[tour[-1]]:
        raise ValueError(
            f"Last visited node {tour[-1]} is not adjacent to node 0; tour does not close."
        )
    tour.append(0)
    return tour
