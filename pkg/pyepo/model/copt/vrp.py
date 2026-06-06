#!/usr/bin/env python
"""
Capacitated vehicle routing problem
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, NoReturn

import numpy as np
from coptpy import COPT, CallbackBase

from pyepo.model.copt.coptmodel import _get_envr, optCoptModel
from pyepo.model.utils import _EDGE_ACTIVE_TOL, _uf_components, unionFind
from pyepo.utils import costToNumpy

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


class vrpABModel(optCoptModel):
    """
    Abstract COPT-backed model for the capacitated vehicle routing problem.

    A single-customer route is excluded so all edge variables stay strictly
    binary; if a single-stop route is actually needed, duplicate the depot.

    Attributes:
        num_nodes: number of nodes (including depot at index 0)
        nodes: list of node indices
        edges: undirected edges as (i, j) with i < j
        demands: customer demands (excluding depot)
        capacity: per-vehicle capacity
        num_vehicle: number of vehicles
    """

    def __init__(
        self,
        num_nodes: int,
        demands: list[float] | np.ndarray,
        capacity: float,
        num_vehicle: int,
    ) -> None:
        """
        Args:
            num_nodes: number of nodes (depot is node 0)
            demands: per-customer demands, length ``num_nodes - 1``
            capacity: vehicle capacity
            num_vehicle: number of vehicles
        """
        # problem parameters
        self.num_nodes = num_nodes
        self.nodes = list(range(num_nodes))
        self.edges = [(i, j) for i in self.nodes for j in self.nodes if i < j]
        self.demands = demands
        self.capacity = capacity
        self.num_vehicle = num_vehicle
        super().__init__()

    @property
    def num_cost(self) -> int:
        """
        number of costs to be predicted
        """
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
        A method to copy the model
        """
        # fresh build to re-populate callback state on the new COPT model
        return type(self)(self.num_nodes, self.demands, self.capacity, self.num_vehicle)


class vrpRCIModel(vrpABModel):
    """
    CVRP formulation with 2-degree constraints and lazy rounded-capacity cuts.

    Uses one undirected Var per edge (``x[j,i]`` aliases ``x[i,j]``). Subtour
    elimination and rounded capacity inequalities are added lazily during
    branch-and-cut via a COPT callback.
    """

    class _RCICallback(CallbackBase):
        """A callback for rounded-capacity / subtour elimination cuts."""

        def __init__(self, x, n, edges, demands, capacity):
            super().__init__()
            self._x = x
            self._n = n
            self._edges = edges
            self._q = {i: demands[i - 1] for i in range(1, n)}
            self._Q = capacity

        def callback(self):
            if self.where() != COPT.CBCONTEXT_MIPSOL:
                return
            # customer-side active edges
            uf = unionFind(self._n)
            for u, v in self._edges:
                if u == 0 or v == 0:
                    continue
                if self.getSolution(self._x[u, v]) > _EDGE_ACTIVE_TOL:
                    uf.union(u, v)
            # rounded-capacity / subtour cut per non-trivial component
            for component in _uf_components(uf):
                if len(component) < 3:
                    continue
                # rounded number of vehicles
                k = int(np.ceil(sum(self._q[v] for v in component) / self._Q))
                # interior edges
                edges_s = [(u, v) for u in component for v in component if u < v]
                if (len(edges_s) >= len(component)) or (k > 1):
                    constr = sum(self._x[e] for e in edges_s) <= len(component) - k
                    self.addLazyConstr(constr)

    def _getModel(self) -> tuple:
        """
        A method to build COPT model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = _get_envr().createModel("vrp")
        # undirected edge variables, with x[j,i] aliasing x[i,j]
        x = m.addVars(self.edges, nameprefix="x", vtype=COPT.BINARY)
        for i, j in self.edges:
            x[j, i] = x[i, j]
        # sense
        m.setObjSense(COPT.MINIMIZE)
        # depot degree
        m.addConstr(sum(x[0, j] for j in self.nodes[1:]) <= 2 * self.num_vehicle)
        # customer 2-degree
        for i in self.nodes[1:]:
            m.addConstr(sum(x[i, j] for j in self.nodes if j != i) == 2)
        # one unique Var per undirected edge (x[j,i] aliases x[i,j])
        self._cost_vars: list = [x[e] for e in self.edges]
        return m, x

    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        """
        A method to set the objective function

        Args:
            c: cost vector aligned with ``self.edges``
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector does not match number of cost variables.")
        c = costToNumpy(c)
        # batch C-level coefficient update
        self._model.setInfo("Obj", self._cost_vars, c.tolist())

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve the model

        Returns:
            tuple: edge-selection vector (uint8) and objective value (float)
        """
        # install lazy callback
        cb = self._RCICallback(
            self.x,
            self.num_nodes,
            self.edges,
            self.demands,
            self.capacity,
        )
        self._model.setCallback(cb, COPT.CBCONTEXT_MIPSOL)
        # optimize
        self._model.solve()
        # threshold to binary selection
        xvals = np.asarray(self._model.getInfo("Value", self._cost_vars))
        sol = (xvals > _EDGE_ACTIVE_TOL).astype(np.uint8)
        return sol, self._model.objVal


class vrpMTZModel(vrpABModel):
    """
    CVRP formulation on a directed graph with MTZ-style capacity constraints
    (no lazy cuts). Cost vector is per undirected edge: cost ``c[k]`` is
    assigned to both ``x[i,j]`` and ``x[j,i]``.
    """

    def _getModel(self) -> tuple:
        """
        A method to build COPT model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = _get_envr().createModel("vrp")
        # directed edge variables (both directions)
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, nameprefix="x", vtype=COPT.BINARY)
        # per-node load auxiliaries with per-customer demand lower bound
        u = m.addVars(self.nodes, nameprefix="u", vtype=COPT.CONTINUOUS, ub=self.capacity)
        for k in self.nodes[1:]:
            u[k].lb = self.demands[k - 1]
        # sense
        m.setObjSense(COPT.MINIMIZE)
        # customer assignment: one in, one out
        for i in self.nodes[1:]:
            m.addConstr(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for j in self.nodes[1:]:
            m.addConstr(sum(x[i, j] for i in self.nodes if i != j) == 1)
        # depot vehicle count (out and in)
        m.addConstr(sum(x[0, j] for j in self.nodes[1:]) <= self.num_vehicle)
        m.addConstr(sum(x[i, 0] for i in self.nodes[1:]) <= self.num_vehicle)
        # MTZ capacity / subtour-free load propagation
        for i, j in directed_edges:
            if i == 0 or j == 0:
                continue
            m.addConstr(
                u[i] - u[j] + self.capacity * x[i, j] <= self.capacity - self.demands[j - 1]
            )
        # cache (x[i,j], x[j,i]) pairs in cost-vector order for batched setInfo/getInfo
        self._cost_vars: list = []
        for i, j in self.edges:
            self._cost_vars.extend([x[i, j], x[j, i]])
        return m, x

    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        """
        A method to set the objective function

        Args:
            c: cost vector aligned with ``self.edges`` (one cost per undirected edge)
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector does not match number of cost variables.")
        c = costToNumpy(c)
        # each undirected edge maps to 2 directed vars; both get coefficient c[k]
        self._model.setInfo("Obj", self._cost_vars, np.repeat(c, 2).tolist())

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve the model

        Returns:
            tuple: edge-selection vector (uint8) and objective value (float)
        """
        # optimize
        self._model.solve()
        # collapse directed pair to undirected selection per edge
        xvals = np.asarray(self._model.getInfo("Value", self._cost_vars)).reshape(-1, 2)
        sol = np.asarray((xvals > _EDGE_ACTIVE_TOL).any(axis=1).astype(np.uint8))
        return sol, self._model.objVal

    def relax(self) -> vrpMTZModelRel:
        """A method to get linear relaxation model"""
        return vrpMTZModelRel(self.num_nodes, self.demands, self.capacity, self.num_vehicle)


class vrpMTZModelRel(vrpMTZModel):
    """LP relaxation of :class:`vrpMTZModel`."""

    def _getModel(self) -> tuple:
        """
        A method to build COPT model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = _get_envr().createModel("vrp")
        # continuous-relaxed directed edge variables
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, nameprefix="x", vtype=COPT.CONTINUOUS, lb=0, ub=1)
        # per-node load auxiliaries with per-customer demand lower bound
        u = m.addVars(self.nodes, nameprefix="u", vtype=COPT.CONTINUOUS, ub=self.capacity)
        for k in self.nodes[1:]:
            u[k].lb = self.demands[k - 1]
        # sense
        m.setObjSense(COPT.MINIMIZE)
        # customer assignment: one in, one out
        for i in self.nodes[1:]:
            m.addConstr(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for j in self.nodes[1:]:
            m.addConstr(sum(x[i, j] for i in self.nodes if i != j) == 1)
        # depot vehicle count (out and in)
        m.addConstr(sum(x[0, j] for j in self.nodes[1:]) <= self.num_vehicle)
        m.addConstr(sum(x[i, 0] for i in self.nodes[1:]) <= self.num_vehicle)
        # MTZ capacity / subtour-free load propagation
        for i, j in directed_edges:
            if i == 0 or j == 0:
                continue
            m.addConstr(
                u[i] - u[j] + self.capacity * x[i, j] <= self.capacity - self.demands[j - 1]
            )
        # cache (x[i,j], x[j,i]) pairs in cost-vector order for batched setInfo/getInfo
        self._cost_vars: list = []
        for i, j in self.edges:
            self._cost_vars.extend([x[i, j], x[j, i]])
        return m, x

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve the model — returns fractional edge selections
        """
        self._model.solve()
        # sum directed pair to per-edge fractional value
        xvals = np.asarray(self._model.getInfo("Value", self._cost_vars)).reshape(-1, 2)
        return xvals.sum(axis=1), self._model.objVal

    def relax(self) -> NoReturn:
        """A forbidden method to relax MIP model"""
        raise RuntimeError("Model has already been relaxed.")

    def getTour(self, sol: np.ndarray | torch.Tensor | list) -> list[list[int]]:
        """A forbidden method to get a tour from solution"""
        raise RuntimeError("Relaxation Model has no integer solution.")
