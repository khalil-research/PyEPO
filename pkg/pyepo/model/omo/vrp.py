#!/usr/bin/env python
"""
Capacitated vehicle routing problem
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from pyepo.model.omo.omomodel import optOmoModel
from pyepo.model.utils import _EDGE_ACTIVE_TOL

if TYPE_CHECKING:
    import torch

try:
    from pyomo import environ as pe
except ImportError:
    pass


class vrpABModel(optOmoModel):
    """
    Abstract Pyomo-backed model for the capacitated vehicle routing problem.

    Pyomo lacks easy callback support, so no lazy-cut RCI formulation exists
    for this backend — only MTZ. A single-customer route is excluded so all
    edge variables stay strictly binary; if a single-stop route is actually
    needed, duplicate the depot.

    Attributes:
        num_nodes: number of nodes (including depot at index 0)
        nodes: list of node indices
        edges: undirected edges as (i, j) with i < j
        demands: customer demands (excluding depot)
        capacity: per-vehicle capacity
        num_vehicle: number of vehicles
        solver: optimization solver in the background
    """

    def __init__(
        self,
        num_nodes: int,
        demands: list[float] | np.ndarray,
        capacity: float,
        num_vehicle: int,
        solver: str = "glpk",
    ) -> None:
        """
        Args:
            num_nodes: number of nodes (depot is node 0)
            demands: per-customer demands, length ``num_nodes - 1``
            capacity: vehicle capacity
            num_vehicle: number of vehicles
            solver: optimization solver in the background
        """
        # problem parameters
        self.num_nodes = num_nodes
        self.nodes = list(range(num_nodes))
        self.edges = [(i, j) for i in self.nodes for j in self.nodes if i < j]
        self.demands = demands
        self.capacity = capacity
        self.num_vehicle = num_vehicle
        super().__init__(solver)

    @property
    def num_cost(self) -> int:
        """
        number of costs to be predicted
        """
        return len(self.edges)

    def _obj_expr(self):
        """Paired: each undirected edge cost weights x[i,j] + x[j,i]."""
        return sum(
            self._model.cost[k] * (self.x[i, j] + self.x[j, i])
            for k, (i, j) in enumerate(self.edges)
        )

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

    def copy(self) -> vrpABModel:
        """
        A method to copy the model
        """
        # fresh build keeps the parameterized objective consistent
        return type(self)(
            self.num_nodes, self.demands, self.capacity, self.num_vehicle, self.solver,
        )


class vrpMTZModel(vrpABModel):
    """
    CVRP formulation on a directed graph with MTZ-style capacity constraints.
    Cost vector is per undirected edge: cost ``c[k]`` is assigned to both
    ``x[i,j]`` and ``x[j,i]``.
    """

    def _getModel(self) -> tuple:
        """
        A method to build Pyomo model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = pe.ConcreteModel("vrp")
        # index sets
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        m.dedges = pe.Set(initialize=directed_edges)
        m.nds = pe.Set(initialize=self.nodes)
        # directed edge variables
        x = pe.Var(m.dedges, domain=pe.Binary)
        m.x = x
        # per-node load auxiliaries with per-customer demand lower bound
        u = pe.Var(m.nds, domain=pe.NonNegativeReals, bounds=(0, self.capacity))
        m.u = u
        for k in self.nodes[1:]:
            u[k].setlb(self.demands[k - 1])
        # constraints
        m.cons = pe.ConstraintList()
        # customer assignment: one in, one out
        for i in self.nodes:
            if i == 0:
                continue
            m.cons.add(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for j in self.nodes:
            if j == 0:
                continue
            m.cons.add(sum(x[i, j] for i in self.nodes if i != j) == 1)
        # depot vehicle count (out and in)
        m.cons.add(sum(x[0, j] for j in self.nodes if j != 0) <= self.num_vehicle)
        m.cons.add(sum(x[i, 0] for i in self.nodes if i != 0) <= self.num_vehicle)
        # MTZ capacity / subtour-free load propagation
        for i, j in directed_edges:
            if i == 0 or j == 0:
                continue
            m.cons.add(
                u[i] - u[j] + self.capacity * x[i, j]
                <= self.capacity - self.demands[j - 1]
            )
        return m, x

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve the model

        Returns:
            tuple: edge-selection vector (uint8) and objective value (float)
        """
        self._solverfac.solve(self._model)
        # collapse directed pair to undirected selection per edge
        sol = np.zeros(self.num_cost, dtype=np.float32)
        for k, (i, j) in enumerate(self.edges):
            if (
                pe.value(self.x[i, j]) > _EDGE_ACTIVE_TOL
                or pe.value(self.x[j, i]) > _EDGE_ACTIVE_TOL
            ):
                sol[k] = 1
        return sol, pe.value(self._model.obj)

    def relax(self) -> vrpMTZModelRel:
        """A method to get linear relaxation model"""
        return vrpMTZModelRel(
            self.num_nodes, self.demands, self.capacity, self.num_vehicle, self.solver,
        )


class vrpMTZModelRel(vrpMTZModel):
    """LP relaxation of :class:`vrpMTZModel`."""

    def _getModel(self) -> tuple:
        """
        A method to build Pyomo model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = pe.ConcreteModel("vrp")
        # index sets
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        m.dedges = pe.Set(initialize=directed_edges)
        m.nds = pe.Set(initialize=self.nodes)
        # continuous-relaxed directed edge variables
        x = pe.Var(m.dedges, domain=pe.NonNegativeReals, bounds=(0, 1))
        m.x = x
        # per-node load auxiliaries with per-customer demand lower bound
        u = pe.Var(m.nds, domain=pe.NonNegativeReals, bounds=(0, self.capacity))
        m.u = u
        for k in self.nodes[1:]:
            u[k].setlb(self.demands[k - 1])
        # constraints
        m.cons = pe.ConstraintList()
        # customer assignment: one in, one out
        for i in self.nodes:
            if i == 0:
                continue
            m.cons.add(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for j in self.nodes:
            if j == 0:
                continue
            m.cons.add(sum(x[i, j] for i in self.nodes if i != j) == 1)
        # depot vehicle count (out and in)
        m.cons.add(sum(x[0, j] for j in self.nodes if j != 0) <= self.num_vehicle)
        m.cons.add(sum(x[i, 0] for i in self.nodes if i != 0) <= self.num_vehicle)
        # MTZ capacity / subtour-free load propagation
        for i, j in directed_edges:
            if i == 0 or j == 0:
                continue
            m.cons.add(
                u[i] - u[j] + self.capacity * x[i, j]
                <= self.capacity - self.demands[j - 1]
            )
        return m, x

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve the model — returns fractional edge selections
        """
        self._solverfac.solve(self._model)
        # sum directed pair to per-edge fractional value
        sol = np.zeros(self.num_cost, dtype=np.float32)
        for k, (i, j) in enumerate(self.edges):
            sol[k] = pe.value(self.x[i, j]) + pe.value(self.x[j, i])
        return sol, pe.value(self._model.obj)

    def relax(self) -> vrpMTZModel:
        """A forbidden method to relax MIP model"""
        raise RuntimeError("Model has already been relaxed.")

    def getTour(self, sol: np.ndarray | torch.Tensor | list) -> list[list[int]]:
        """A forbidden method to get a tour from solution"""
        raise RuntimeError("Relaxation Model has no integer solution.")
