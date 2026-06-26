#!/usr/bin/env python
"""
Capacitated vehicle routing problem with binding-constraint tracking for CaVE
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, NoReturn

import numpy as np

with contextlib.suppress(ImportError):
    import gurobipy as gp
    from gurobipy import GRB

from pyepo.model._common import validate_objective_shape
from pyepo.model.bases import vrpABBase
from pyepo.model.grb.grbmodel import _promote_lazy_cuts, _require_solution, optGrbModel
from pyepo.model.utils import _EDGE_ACTIVE_TOL, _uf_components, unionFind
from pyepo.utils import costToNumpy

if TYPE_CHECKING:
    import torch


class vrpABModel(vrpABBase, optGrbModel):
    """
    Abstract Gurobi-backed model for the capacitated vehicle routing problem.

    A single-customer route is excluded so all edge variables stay strictly
    binary; if a single-stop route is actually needed, duplicate the depot.
    """

    def _addExtraConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> None:
        # coefs @ edge-selection <= rhs over the cost-aligned vars
        expr = gp.LinExpr(self._expand_coefs(np.asarray(coefs)).tolist(), self._cost_vars)
        self._model.addConstr(expr <= rhs)


class vrpRCIModel(vrpABModel):
    """
    CVRP formulation with 2-degree constraints and lazy rounded-capacity cuts.

    Uses one undirected Var per edge (``x[j,i]`` aliases ``x[i,j]``). Subtour
    elimination and rounded capacity inequalities are added lazily during
    branch-and-cut; the cuts added at the optimum are tracked on
    ``model._lazy_constrs`` for downstream CaVE cone extraction. With
    ``recycle_cuts`` the cuts found in one solve join the model's lazy pool,
    so later solves on new cost vectors skip rediscovering them.
    """

    def __init__(
        self,
        num_nodes: int,
        demands: list[float] | np.ndarray,
        capacity: float,
        num_vehicle: int,
        recycle_cuts: bool = False,
    ) -> None:
        """
        Args:
            num_nodes: number of nodes (depot is node 0)
            demands: per-customer demands, length ``num_nodes - 1``
            capacity: vehicle capacity
            num_vehicle: number of vehicles
            recycle_cuts: keep generated cuts in the lazy pool across solves
        """
        self.recycle_cuts = recycle_cuts
        self._recycled_keys: set = set()
        super().__init__(num_nodes, demands, capacity, num_vehicle)

    def get_config(self) -> dict:
        return {**super().get_config(), "recycle_cuts": self.recycle_cuts}

    def _getModel(self) -> tuple:
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = gp.Model("vrp")
        # silence param-setting output before any setParam call
        m.Params.outputFlag = 0
        # undirected edge variables, with x[j,i] aliasing x[i,j]
        x = m.addVars(self.edges, name="x", vtype=GRB.BINARY)
        for i, j in self.edges:
            x[j, i] = x[i, j]
        # sense
        m.modelSense = GRB.MINIMIZE
        # depot degree
        m.addConstr(x.sum(0, "*") <= 2 * self.num_vehicle)  # 2 per vehicle
        # customer 2-degree
        m.addConstrs(x.sum(i, "*") == 2 for i in self.nodes[1:])
        # callback state
        m._x = x
        m._n = self.num_nodes
        m._q = {i: self.demands[i - 1] for i in self.nodes[1:]}
        m._Q = self.capacity
        m._edges = self.edges
        m._lazy_constrs = []
        # activate lazy constraints
        m.Params.lazyConstraints = 1
        # one unique Var per undirected edge (x[j,i] aliases x[i,j])
        self._cost_vars: list = [x[e] for e in self.edges]
        return m, x

    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        """
        A method to set the objective function

        Args:
            c: cost vector aligned with ``self.edges``
        """
        validate_objective_shape(c, self.num_cost)
        c = costToNumpy(c)
        # batch C-level coefficient update
        self._model.setAttr("Obj", self._cost_vars, c.tolist())

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve the model

        Returns:
            tuple: edge-selection vector (uint8) and objective value (float)
        """
        # promote the previous solve's cuts into the lazy pool
        if self.recycle_cuts:
            _promote_lazy_cuts(self._model, self._recycled_keys)
        # the cut buffer tracks the current solve only
        self._model._lazy_constrs = []
        # optimize
        self._model.optimize(self._vrp_callback)
        _require_solution(self._model)
        # threshold to binary selection
        xvals = np.asarray(self._model.getAttr("X", self._cost_vars))
        sol = (xvals > _EDGE_ACTIVE_TOL).astype(np.uint8)
        return sol, self._model.objVal

    @staticmethod
    def _vrp_callback(model, where):
        """
        A static method to add lazy constraints for rounded capacity / subtour
        """
        if where != GRB.Callback.MIPSOL:
            return
        # customer-side active edges
        uf = unionFind(model._n)
        for u, v in model._edges:
            if u == 0 or v == 0:
                continue
            if model.cbGetSolution(model._x[u, v]) > _EDGE_ACTIVE_TOL:
                uf.union(u, v)
        # rounded-capacity / subtour cut per non-trivial component
        for component in _uf_components(uf):
            if len(component) < 2:
                continue
            # rounded number of vehicles
            k = int(np.ceil(sum(model._q[v] for v in component) / model._Q))
            # interior edges
            edges_s = [(u, v) for u in component for v in component if u < v]
            if (len(edges_s) >= len(component)) or (k > 1):
                constr = gp.quicksum(model._x[e] for e in edges_s) <= len(component) - k
                model.cbLazy(constr)
                # track for downstream binding-constraint extraction
                model._lazy_constrs.append(constr)


class vrpMTZModel(vrpABModel):
    """
    CVRP formulation on a directed graph with MTZ-style capacity constraints
    (no lazy cuts). Cost vector is per undirected edge: cost ``c[k]`` is
    assigned to both ``x[i,j]`` and ``x[j,i]``.
    """

    def _expand_coefs(self, coefs: np.ndarray) -> np.ndarray:
        # each undirected edge maps to 2 directed cost vars
        return np.repeat(coefs, 2)

    def _getModel(self) -> tuple:
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = gp.Model("vrp")
        # directed edge variables (both directions) + per-node load auxiliaries
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, name="x", vtype=GRB.BINARY)
        u = m.addVars(
            self.nodes,
            name="u",
            lb=[0, *list(self.demands)],
            ub=self.capacity,
            vtype=GRB.CONTINUOUS,
        )
        # sense
        m.modelSense = GRB.MINIMIZE
        # customer assignment: one in, one out
        m.addConstrs(
            gp.quicksum(x[i, j] for j in self.nodes if j != i) == 1 for i in self.nodes[1:]
        )
        m.addConstrs(
            gp.quicksum(x[i, j] for i in self.nodes if i != j) == 1 for j in self.nodes[1:]
        )
        # depot vehicle count (out and in)
        m.addConstr(x.sum(0, "*") <= self.num_vehicle)
        m.addConstr(x.sum("*", 0) <= self.num_vehicle)
        # MTZ capacity / subtour-free load propagation
        m.addConstrs(
            u[i] - u[j] + self.capacity * x[i, j] <= self.capacity - self.demands[j - 1]
            for i, j in directed_edges
            if i != 0 and j != 0
        )
        # cache (x[i,j], x[j,i]) pairs in cost-vector order for batched setAttr/getAttr
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
        validate_objective_shape(c, self.num_cost)
        c = costToNumpy(c)
        # each undirected edge maps to 2 directed vars; both get coefficient c[k]
        self._model.setAttr("Obj", self._cost_vars, np.repeat(c, 2).tolist())

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve the model

        Returns:
            tuple: edge-selection vector (uint8) and objective value (float)
        """
        # optimize
        self._model.optimize()
        _require_solution(self._model)
        # collapse directed pair to undirected selection per edge
        xvals = np.asarray(self._model.getAttr("X", self._cost_vars)).reshape(-1, 2)
        sol = np.asarray((xvals > _EDGE_ACTIVE_TOL).any(axis=1).astype(np.uint8))
        return sol, self._model.objVal

    def relax(self) -> vrpMTZModelRel:
        """A method to get linear relaxation model"""
        model_rel = vrpMTZModelRel(self.num_nodes, self.demands, self.capacity, self.num_vehicle)
        # replay user cuts on the relaxation
        self._replay_extras(model_rel)
        return model_rel


class vrpMTZModelRel(vrpMTZModel):
    """LP relaxation of :class:`vrpMTZModel`."""

    def _getModel(self) -> tuple:
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = gp.Model("vrp")
        # continuous-relaxed directed edge variables + per-node load auxiliaries
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, name="x", vtype=GRB.CONTINUOUS)
        u = m.addVars(
            self.nodes,
            name="u",
            lb=[0, *list(self.demands)],
            ub=self.capacity,
            vtype=GRB.CONTINUOUS,
        )
        # sense
        m.modelSense = GRB.MINIMIZE
        # customer assignment: one in, one out
        m.addConstrs(
            gp.quicksum(x[i, j] for j in self.nodes if j != i) == 1 for i in self.nodes[1:]
        )
        m.addConstrs(
            gp.quicksum(x[i, j] for i in self.nodes if i != j) == 1 for j in self.nodes[1:]
        )
        # depot vehicle count (out and in)
        m.addConstr(x.sum(0, "*") <= self.num_vehicle)
        m.addConstr(x.sum("*", 0) <= self.num_vehicle)
        # MTZ capacity / subtour-free load propagation
        m.addConstrs(
            u[i] - u[j] + self.capacity * x[i, j] <= self.capacity - self.demands[j - 1]
            for i, j in directed_edges
            if i != 0 and j != 0
        )
        # cache (x[i,j], x[j,i]) pairs in cost-vector order for batched setAttr/getAttr
        self._cost_vars: list = []
        for i, j in self.edges:
            self._cost_vars.extend([x[i, j], x[j, i]])
        return m, x

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve the model — returns fractional edge selections
        """
        self._model.optimize()
        _require_solution(self._model)
        # sum directed pair to per-edge fractional value
        xvals = np.asarray(self._model.getAttr("X", self._cost_vars)).reshape(-1, 2)
        return xvals.sum(axis=1), self._model.objVal

    def relax(self) -> NoReturn:
        """A forbidden method to relax MIP model"""
        raise RuntimeError("Model has already been relaxed.")

    def getTour(self, sol: np.ndarray | torch.Tensor | list) -> list[list[int]]:
        """A forbidden method to get a tour from solution"""
        raise RuntimeError("Relaxation Model has no integer solution.")
