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
    threshold: float = 1e-2,
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
            "the model returned an infeasible TSP solution."
        )
    # walk the tour starting from the first node with an active edge
    start = next(iter(edges))
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
                "tour. Check that subtour elimination constraints are active."
            )
    if 0 in edges[tour[-1]]:
        tour.append(0)
    return tour
