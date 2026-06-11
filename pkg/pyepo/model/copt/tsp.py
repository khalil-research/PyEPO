#!/usr/bin/env python
"""
Traveling salesman problem
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, NoReturn

import numpy as np

try:
    from coptpy import COPT, CallbackBase, LinExpr
except ImportError:
    CallbackBase = object  # placeholder so class bodies evaluate without coptpy

from pyepo.model.bases import tspABBase
from pyepo.model.copt.coptmodel import _get_envr, optCoptModel
from pyepo.model.utils import _EDGE_ACTIVE_TOL, unionFind
from pyepo.utils import costToNumpy

if TYPE_CHECKING:
    import torch


class tspABModel(tspABBase, optCoptModel):
    """
    COPT-backed TSP abstract base. Provides paired-variable ``setObj`` /
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
        self._model.setInfo("Obj", self._cost_vars, np.repeat(c, 2).tolist())

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve model
        """
        self._model.solve()
        xvals = np.asarray(self._model.getInfo("Value", self._cost_vars)).reshape(-1, 2)
        sol = np.asarray((xvals > _EDGE_ACTIVE_TOL).any(axis=1).astype(np.uint8))
        return sol, self._model.objVal

    def _addExtraConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> None:
        """Add a single linear constraint to ``self._model`` using paired (x[i,j] + x[j,i])."""
        # both directed Vars share coef[k]; mirror setObj's np.repeat layout
        expr = LinExpr()
        expr.addTerms(self._cost_vars, np.repeat(coefs, 2).tolist())
        self._model.addConstr(expr <= rhs)


class tspGGModel(tspABModel):
    """
    Gavish-Graves (GG) formulation.
    """

    def _getModel(self) -> tuple:
        """
        A method to build COPT model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = _get_envr().createModel("tsp")
        # variables
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, nameprefix="x", vtype=COPT.BINARY)
        y = m.addVars(directed_edges, nameprefix="y", vtype=COPT.CONTINUOUS)
        # sense
        m.setObjSense(COPT.MINIMIZE)
        # constraints
        for j in self.nodes:
            m.addConstr(sum(x[i, j] for i in self.nodes if i != j) == 1)
        for i in self.nodes:
            m.addConstr(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for i in self.nodes[1:]:
            m.addConstr(
                sum(y[i, j] for j in self.nodes if j != i)
                - sum(y[j, i] for j in self.nodes[1:] if j != i)
                == 1
            )
        for i, j in directed_edges:
            if i != 0:
                m.addConstr(y[i, j] <= (len(self.nodes) - 1) * x[i, j])
        # cache (x[i,j], x[j,i]) pairs in cost-vector order for batch setInfo/getInfo
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
        A method to build COPT model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = _get_envr().createModel("tsp")
        # variables
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, nameprefix="x", vtype=COPT.CONTINUOUS, lb=0, ub=1)
        y = m.addVars(directed_edges, nameprefix="y", vtype=COPT.CONTINUOUS)
        # sense
        m.setObjSense(COPT.MINIMIZE)
        # constraints
        for j in self.nodes:
            m.addConstr(sum(x[i, j] for i in self.nodes if i != j) == 1)
        for i in self.nodes:
            m.addConstr(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for i in self.nodes[1:]:
            m.addConstr(
                sum(y[i, j] for j in self.nodes if j != i)
                - sum(y[j, i] for j in self.nodes[1:] if j != i)
                == 1
            )
        for i, j in directed_edges:
            if i != 0:
                m.addConstr(y[i, j] <= (len(self.nodes) - 1) * x[i, j])
        # cache (x[i,j], x[j,i]) pairs in cost-vector order for batch setInfo/getInfo
        self._cost_vars: list = []
        for i, j in self.edges:
            self._cost_vars.extend([x[i, j], x[j, i]])
        return m, x

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve model — returns fractional solution.
        """
        self._model.solve()
        xvals = np.asarray(self._model.getInfo("Value", self._cost_vars)).reshape(-1, 2)
        sol = xvals.sum(axis=1)
        return sol, self._model.objVal

    def relax(self) -> NoReturn:
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

    Uses one undirected Var per edge, so the paired-vars ``setObj`` /
    ``solve`` / ``_addExtraConstr`` inherited from ``tspABModel`` are
    overridden here.
    """

    class _SubtourCallback(CallbackBase):
        """
        A callback class for subtour elimination
        """

        def __init__(self, x, n, edges):
            super().__init__()
            self._x = x
            self._n = n
            self._edges = edges

        def callback(self):
            if self.where() == COPT.CBCONTEXT_MIPSOL:
                # selected edges
                xvals = self.getSolution(self._x)
                selected = [(i, j) for i, j in self._x if xvals[i, j] > _EDGE_ACTIVE_TOL]
                # check subcycle with unionfind
                uf = unionFind(self._n)
                for i, j in selected:
                    if not uf.union(i, j):
                        # find subcycle
                        cycle = [k for k in range(self._n) if uf.find(k) == uf.find(i)]
                        if len(cycle) < self._n:
                            constr = sum(self._x[i, j] for i, j in combinations(cycle, 2))
                            self.addLazyConstr(constr <= len(cycle) - 1)
                        break

    def _getModel(self) -> tuple:
        """
        A method to build COPT model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = _get_envr().createModel("tsp")
        # variables
        x = m.addVars(self.edges, nameprefix="x", vtype=COPT.BINARY)
        for i, j in self.edges:
            x[j, i] = x[i, j]
        # sense
        m.setObjSense(COPT.MINIMIZE)
        # constraints
        for i in self.nodes:
            m.addConstr(sum(x[i, j] for j in self.nodes if j != i) == 2)
        # one unique Var per undirected edge (x[j,i] aliases x[i,j])
        self._cost_vars: list = [x[e] for e in self.edges]
        return m, x

    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        """
        A method to set the objective function

        Args:
            c: cost vector
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector does not match number of cost variables.")
        c = costToNumpy(c)
        self._model.setInfo("Obj", self._cost_vars, c.tolist())

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve model
        """
        cb = self._SubtourCallback(self.x, len(self.nodes), self.edges)
        self._model.setCallback(cb, COPT.CBCONTEXT_MIPSOL)
        self._model.solve()
        xvals = np.asarray(self._model.getInfo("Value", self._cost_vars))
        sol = (xvals > _EDGE_ACTIVE_TOL).astype(np.uint8)
        return sol, self._model.objVal

    def _addExtraConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> None:
        """Add a single linear constraint to ``self._model`` using the DFJ variable scheme."""
        expr = LinExpr()
        expr.addTerms(self._cost_vars, coefs.tolist())
        self._model.addConstr(expr <= rhs)


class tspMTZModel(tspABModel):
    """
    Miller-Tucker-Zemlin (MTZ) formulation.
    """

    def _getModel(self) -> tuple:
        """
        A method to build COPT model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = _get_envr().createModel("tsp")
        # variables
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, nameprefix="x", vtype=COPT.BINARY)
        u = m.addVars(self.nodes, nameprefix="u", vtype=COPT.CONTINUOUS)
        # sense
        m.setObjSense(COPT.MINIMIZE)
        # constraints
        for j in self.nodes:
            m.addConstr(sum(x[i, j] for i in self.nodes if i != j) == 1)
        for i in self.nodes:
            m.addConstr(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for i, j in directed_edges:
            if (i != 0) and (j != 0):
                m.addConstr(u[j] - u[i] >= len(self.nodes) * (x[i, j] - 1) + 1)
        # cache (x[i,j], x[j,i]) pairs in cost-vector order for batch setInfo/getInfo
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
        A method to build COPT model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = _get_envr().createModel("tsp")
        # variables
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, nameprefix="x", vtype=COPT.CONTINUOUS, lb=0, ub=1)
        u = m.addVars(self.nodes, nameprefix="u", vtype=COPT.CONTINUOUS)
        # sense
        m.setObjSense(COPT.MINIMIZE)
        # constraints
        for j in self.nodes:
            m.addConstr(sum(x[i, j] for i in self.nodes if i != j) == 1)
        for i in self.nodes:
            m.addConstr(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for i, j in directed_edges:
            if (i != 0) and (j != 0):
                m.addConstr(u[j] - u[i] >= len(self.nodes) * (x[i, j] - 1) + 1)
        # cache (x[i,j], x[j,i]) pairs in cost-vector order for batch setInfo/getInfo
        self._cost_vars: list = []
        for i, j in self.edges:
            self._cost_vars.extend([x[i, j], x[j, i]])
        return m, x

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve model — returns fractional solution.
        """
        self._model.solve()
        xvals = np.asarray(self._model.getInfo("Value", self._cost_vars)).reshape(-1, 2)
        sol = xvals.sum(axis=1)
        return sol, self._model.objVal

    def relax(self) -> NoReturn:
        """
        A forbidden method to relax MIP model
        """
        raise RuntimeError("Model has already been relaxed.")

    def getTour(self, sol: np.ndarray | torch.Tensor | list) -> list[int]:
        """
        A forbidden method to get a tour from solution
        """
        raise RuntimeError("Relaxation Model has no integer solution.")


if __name__ == "__main__":
    import random

    # random seed
    random.seed(42)
    num_nodes = 5
    num_edges = num_nodes * (num_nodes - 1) // 2
    cost = [random.random() for _ in range(num_edges)]

    # solve GG model
    optmodel = tspGGModel(num_nodes=num_nodes)
    optmodel = optmodel.copy()
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print(f"GG Obj: {obj}")
    tour = optmodel.getTour(sol)
    print(f"GG Tour: {tour}")

    # solve DFJ model
    optmodel = tspDFJModel(num_nodes=num_nodes)
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print(f"DFJ Obj: {obj}")
    tour = optmodel.getTour(sol)
    print(f"DFJ Tour: {tour}")

    # solve MTZ model
    optmodel = tspMTZModel(num_nodes=num_nodes)
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print(f"MTZ Obj: {obj}")
    tour = optmodel.getTour(sol)
    print(f"MTZ Tour: {tour}")

    # relax GG model
    optmodel = tspGGModel(num_nodes=num_nodes)
    optmodel = optmodel.relax()
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print(f"GG Relaxed Obj: {obj}")

    # add constraint
    optmodel = tspMTZModel(num_nodes=num_nodes)
    optmodel = optmodel.addConstr([1] * num_edges, num_edges - 1)
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print(f"MTZ + Constr Obj: {obj}")
