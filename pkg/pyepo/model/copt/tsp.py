#!/usr/bin/env python
"""
Traveling salesman problem
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from coptpy import COPT, CallbackBase, Envr

from pyepo.model.copt.coptmodel import optCoptModel
from pyepo.model.utils import getTspTour, unionFind
from pyepo.utils import costToNumpy

if TYPE_CHECKING:
    import torch


class tspABModel(optCoptModel):
    """
    This abstract class is an optimization model for the traveling salesman problem.
    This model is for further implementation of different formulation.

    Attributes:
        _model (COPT model): COPT model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def __init__(self, num_nodes: int) -> None:
        """
        Args:
            num_nodes: number of nodes
        """
        self.num_nodes = num_nodes
        self.nodes = list(range(num_nodes))
        self.edges = [(i, j) for i in self.nodes for j in self.nodes if i < j]
        super().__init__()
        # constraints added via addConstr, replayed on copy/relax
        self._extra_constrs = []

    @property
    def num_cost(self) -> int:
        return len(self.edges)

    def copy(self) -> tspABModel:
        """
        A method to copy model

        Returns:
            optModel: new copied model
        """
        new_model = type(self)(self.num_nodes)
        self._replay_extras(new_model)
        return new_model

    def _replay_extras(self, other: tspABModel) -> None:
        """Replay self._extra_constrs onto another TSP model of compatible formulation."""
        for coefs, rhs in self._extra_constrs:
            other._extra_constrs.append((coefs, rhs))
            other._addExtraConstr(coefs, rhs)

    def getTour(self, sol: np.ndarray | torch.Tensor | list) -> list[int]:
        """
        A method to get a tour from solution

        Args:
            sol: solution

        Returns:
            list: a TSP tour

        Raises:
            ValueError: if the solution does not form a single connected tour.
        """
        return getTspTour(self.edges, self.num_nodes, sol)


class tspGGModel(tspABModel):
    """
    This class is an optimization model for the traveling salesman problem based on Gavish-Graves (GG) formulation.

    Attributes:
        _model (COPT model): COPT model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self) -> tuple:
        """
        A method to build COPT model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = Envr().createModel("tsp")
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
        sol = (xvals > 1e-2).any(axis=1).astype(np.uint8)
        return sol, self._model.objVal

    def _addExtraConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> None:
        """Add a single linear constraint to self._model using the GG variable scheme."""
        self._model.addConstr(
            sum(coefs[k] * (self.x[i, j] + self.x[j, i]) for k, (i, j) in enumerate(self.edges))
            <= rhs
        )

    def addConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> tspABModel:
        """
        A method to add new constraint

        Args:
            coefs: coefficients of new constraint
            rhs: right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        if len(coefs) != self.num_cost:
            raise ValueError("Size of coef vector does not match number of cost variables.")
        new_model = self.copy()
        new_model._extra_constrs.append((coefs, rhs))
        new_model._addExtraConstr(coefs, rhs)
        return new_model

    def relax(self) -> tspGGModelRel:
        """
        A method to get linear relaxation model
        """
        model_rel = tspGGModelRel(self.num_nodes)
        self._replay_extras(model_rel)
        return model_rel


class tspGGModelRel(tspGGModel):
    """
    This class is relaxation of tspGGModel.

    Attributes:
        _model (COPT model): COPT model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self) -> tuple:
        """
        A method to build COPT model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = Envr().createModel("tsp")
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
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._model.solve()
        xvals = np.asarray(self._model.getInfo("Value", self._cost_vars)).reshape(-1, 2)
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
    This class is an optimization model for the traveling salesman problem based on Danzig-Fulkerson-Johnson (DFJ) formulation and
    constraint generation.

    Attributes:
        _model (COPT model): COPT model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
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
                selected = [(i, j) for i, j in self._x if xvals[i, j] > 1e-2]
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
        m = Envr().createModel("tsp")
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
        sol = (xvals > 1e-2).astype(np.uint8)
        return sol, self._model.objVal

    def _addExtraConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> None:
        """Add a single linear constraint to self._model using the DFJ variable scheme."""
        self._model.addConstr(sum(coefs[i] * self.x[k] for i, k in enumerate(self.edges)) <= rhs)

    def addConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> tspABModel:
        """
        A method to add new constraint

        Args:
            coefs: coefficients of new constraint
            rhs: right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        if len(coefs) != self.num_cost:
            raise ValueError("Size of coef vector does not match number of cost variables.")
        new_model = self.copy()
        new_model._extra_constrs.append((coefs, rhs))
        new_model._addExtraConstr(coefs, rhs)
        return new_model


class tspMTZModel(tspABModel):
    """
    This class is an optimization model for the traveling salesman problem based on Miller-Tucker-Zemlin (MTZ) formulation.

    Attributes:
        _model (COPT model): COPT model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self) -> tuple:
        """
        A method to build COPT model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = Envr().createModel("tsp")
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
        sol = (xvals > 1e-2).any(axis=1).astype(np.uint8)
        return sol, self._model.objVal

    def _addExtraConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> None:
        """Add a single linear constraint to self._model using the MTZ variable scheme."""
        self._model.addConstr(
            sum(coefs[k] * (self.x[i, j] + self.x[j, i]) for k, (i, j) in enumerate(self.edges))
            <= rhs
        )

    def addConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> tspABModel:
        """
        A method to add new constraint

        Args:
            coefs: coefficients of new constraint
            rhs: right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        if len(coefs) != self.num_cost:
            raise ValueError("Size of coef vector does not match number of cost variables.")
        new_model = self.copy()
        new_model._extra_constrs.append((coefs, rhs))
        new_model._addExtraConstr(coefs, rhs)
        return new_model

    def relax(self) -> tspMTZModelRel:
        """
        A method to get linear relaxation model
        """
        model_rel = tspMTZModelRel(self.num_nodes)
        self._replay_extras(model_rel)
        return model_rel


class tspMTZModelRel(tspMTZModel):
    """
    This class is relaxation of tspMTZModel.

    Attributes:
        _model (COPT model): COPT model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self) -> tuple:
        """
        A method to build COPT model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = Envr().createModel("tsp")
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
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._model.solve()
        xvals = np.asarray(self._model.getInfo("Value", self._cost_vars)).reshape(-1, 2)
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
