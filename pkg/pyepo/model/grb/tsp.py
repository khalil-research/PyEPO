#!/usr/bin/env python
"""
Traveling salesman problem
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    pass

from pyepo.model.bases import tspABBase
from pyepo.model.grb.grbmodel import optGrbModel
from pyepo.model.utils import _EDGE_ACTIVE_TOL, unionFind
from pyepo.utils import costToNumpy

if TYPE_CHECKING:
    import torch


class tspABModel(tspABBase, optGrbModel):
    """
    Gurobi-backed TSP abstract base. Provides paired-variable ``setObj`` /
    ``solve`` / ``_addExtraConstr`` shared by GG and MTZ. DFJ overrides those.
    """

    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        """
        A method to set the objective function

        Args:
            c: cost vector
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector does not match number of cost variables.")
        c = costToNumpy(c)
        # each undirected edge maps to 2 directed Vars; both get coefficient c[k]
        self._model.setAttr("Obj", self._cost_vars, np.repeat(c, 2).tolist())

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve model
        """
        self._model.optimize()
        xvals = np.asarray(self._model.getAttr("X", self._cost_vars)).reshape(-1, 2)
        sol = (xvals > _EDGE_ACTIVE_TOL).any(axis=1).astype(np.uint8)
        return sol, self._model.objVal

    def _addExtraConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> None:
        """Add a single linear constraint to ``self._model`` using paired (x[i,j] + x[j,i])."""
        self._model.addConstr(
            gp.quicksum(
                coefs[k] * (self.x[i, j] + self.x[j, i]) for k, (i, j) in enumerate(self.edges)
            )
            <= rhs
        )


class tspGGModel(tspABModel):
    """
    Gavish-Graves (GG) formulation.
    """

    def _getModel(self) -> tuple:
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = gp.Model("tsp")
        # variables
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, name="x", vtype=GRB.BINARY)
        y = m.addVars(directed_edges, name="y")
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstrs(x.sum("*", j) == 1 for j in self.nodes)
        m.addConstrs(x.sum(i, "*") == 1 for i in self.nodes)
        m.addConstrs(
            y.sum(i, "*") - gp.quicksum(y[j, i] for j in self.nodes[1:] if j != i) == 1
            for i in self.nodes[1:]
        )
        m.addConstrs(y[i, j] <= (len(self.nodes) - 1) * x[i, j] for (i, j) in x if i != 0)
        # cache (x[i,j], x[j,i]) pairs in cost-vector order for batch setAttr/getAttr
        self._cost_vars: list = []
        for i, j in self.edges:
            self._cost_vars.extend([x[i, j], x[j, i]])
        return m, x

    def relax(self) -> tspGGModelRel:
        """
        A method to get linear relaxation model
        """
        model_rel = tspGGModelRel(self.num_nodes)
        self._replay_extras(model_rel)
        return model_rel


class tspGGModelRel(tspGGModel):
    """
    LP relaxation of the GG formulation.
    """

    def _getModel(self) -> tuple:
        """
        A method to build Gurobi model
        """
        # create a model
        m = gp.Model("tsp")
        # variables
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, name="x", ub=1)
        y = m.addVars(directed_edges, name="y")
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstrs(x.sum("*", j) == 1 for j in self.nodes)
        m.addConstrs(x.sum(i, "*") == 1 for i in self.nodes)
        m.addConstrs(
            y.sum(i, "*") - gp.quicksum(y[j, i] for j in self.nodes[1:] if j != i) == 1
            for i in self.nodes[1:]
        )
        m.addConstrs(y[i, j] <= (len(self.nodes) - 1) * x[i, j] for (i, j) in x if i != 0)
        # cache (x[i,j], x[j,i]) pairs in cost-vector order for batch setAttr/getAttr
        self._cost_vars: list = []
        for i, j in self.edges:
            self._cost_vars.extend([x[i, j], x[j, i]])
        return m, x

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve model — returns fractional solution.
        """
        self._model.optimize()
        xvals = np.asarray(self._model.getAttr("X", self._cost_vars)).reshape(-1, 2)
        sol = xvals.sum(axis=1)
        return sol, self._model.objVal

    def relax(self) -> tspABModel:
        """
        A forbidden method to relax MIP model
        """
        raise RuntimeError("Model has already been relaxed.")

    def getTour(self, sol: np.ndarray | torch.Tensor | list) -> list[int]:
        """
        A forbidden method to get a tour from solution
        """
        raise RuntimeError("Relaxation Model has no integer solution.")


class tspDFJModel(tspABModel):
    """
    Danzig-Fulkerson-Johnson (DFJ) formulation with lazy subtour elimination.

    Uses one undirected Var per edge (``x[j,i]`` aliases ``x[i,j]``), so the
    paired-vars ``setObj`` / ``solve`` / ``_addExtraConstr`` inherited from
    ``tspABModel`` are overridden here.
    """

    def _getModel(self) -> tuple:
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = gp.Model("tsp")
        # variables
        x = m.addVars(self.edges, name="x", vtype=GRB.BINARY)
        for i, j in self.edges:
            x[j, i] = x[i, j]
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstrs(x.sum(i, "*") == 2 for i in self.nodes)  # 2 degree
        # activate lazy constraints
        m._x = x
        m._n = len(self.nodes)
        m.Params.lazyConstraints = 1
        # one unique Var per undirected edge (x[j,i] aliases x[i,j])
        self._cost_vars: list = [x[e] for e in self.edges]
        return m, x

    @staticmethod
    def _subtourelim(model, where):
        """
        A static method to add lazy constraints for subtour elimination
        """
        if where == GRB.Callback.MIPSOL:
            # selected edges
            xvals = model.cbGetSolution(model._x)
            selected = gp.tuplelist(
                (i, j) for i, j in model._x if xvals[i, j] > _EDGE_ACTIVE_TOL
            )
            # check subcycle with unionfind
            uf = unionFind(model._n)
            for i, j in selected:
                if not uf.union(i, j):
                    # find subcycle
                    cycle = [k for k in range(model._n) if uf.find(k) == uf.find(i)]
                    if len(cycle) < model._n:
                        constr = (
                            gp.quicksum(model._x[i, j] for i, j in combinations(cycle, 2))
                            <= len(cycle) - 1
                        )
                        model.cbLazy(constr)
                    break

    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        """
        A method to set the objective function

        Args:
            c: cost vector
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector does not match number of cost variables.")
        c = costToNumpy(c)
        self._model.setAttr("Obj", self._cost_vars, c.tolist())

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve model
        """
        self._model.optimize(self._subtourelim)
        xvals = np.asarray(self._model.getAttr("X", self._cost_vars))
        sol = (xvals > _EDGE_ACTIVE_TOL).astype(np.uint8)
        return sol, self._model.objVal

    def _addExtraConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> None:
        """Add a single linear constraint to ``self._model`` using the DFJ variable scheme."""
        self._model.addConstr(
            gp.quicksum(coefs[i] * self.x[k] for i, k in enumerate(self.edges)) <= rhs
        )


class tspMTZModel(tspABModel):
    """
    Miller-Tucker-Zemlin (MTZ) formulation.
    """

    def _getModel(self) -> tuple:
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = gp.Model("tsp")
        # variables
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, name="x", vtype=GRB.BINARY)
        u = m.addVars(self.nodes, name="u")
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstrs(x.sum("*", j) == 1 for j in self.nodes)
        m.addConstrs(x.sum(i, "*") == 1 for i in self.nodes)
        m.addConstrs(
            u[j] - u[i] >= len(self.nodes) * (x[i, j] - 1) + 1
            for (i, j) in directed_edges
            if (i != 0) and (j != 0)
        )
        # cache (x[i,j], x[j,i]) pairs in cost-vector order for batch setAttr/getAttr
        self._cost_vars: list = []
        for i, j in self.edges:
            self._cost_vars.extend([x[i, j], x[j, i]])
        return m, x

    def relax(self) -> tspMTZModelRel:
        """
        A method to get linear relaxation model
        """
        model_rel = tspMTZModelRel(self.num_nodes)
        self._replay_extras(model_rel)
        return model_rel


class tspMTZModelRel(tspMTZModel):
    """
    LP relaxation of the MTZ formulation.
    """

    def _getModel(self) -> tuple:
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = gp.Model("tsp")
        # variables
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, name="x", ub=1)
        u = m.addVars(self.nodes, name="u")
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstrs(x.sum("*", j) == 1 for j in self.nodes)
        m.addConstrs(x.sum(i, "*") == 1 for i in self.nodes)
        m.addConstrs(
            u[j] - u[i] >= len(self.nodes) * (x[i, j] - 1) + 1
            for (i, j) in directed_edges
            if (i != 0) and (j != 0)
        )
        # cache (x[i,j], x[j,i]) pairs in cost-vector order for batch setAttr/getAttr
        self._cost_vars: list = []
        for i, j in self.edges:
            self._cost_vars.extend([x[i, j], x[j, i]])
        return m, x

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve model — returns fractional solution.
        """
        self._model.optimize()
        xvals = np.asarray(self._model.getAttr("X", self._cost_vars)).reshape(-1, 2)
        sol = xvals.sum(axis=1)
        return sol, self._model.objVal

    def relax(self) -> tspABModel:
        """
        A forbidden method to relax MIP model
        """
        raise RuntimeError("Model has already been relaxed.")

    def getTour(self, sol: np.ndarray | torch.Tensor | list) -> list[int]:
        """
        A forbidden method to get a tour from solution
        """
        raise RuntimeError("Relaxation Model has no integer solution.")
