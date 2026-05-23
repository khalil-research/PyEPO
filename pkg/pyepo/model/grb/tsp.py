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

    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

from pyepo.model.grb.grbmodel import optGrbModel
from pyepo.model.opt import getTspTour, unionFind

if TYPE_CHECKING:
    import torch


class tspABModel(optGrbModel):
    """
    This abstract class is an optimization model for the traveling salesman problem.
    This model is for further implementation of different formulation.

    Attributes:
        _model (GurobiPy model): Gurobi model
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
    This class is an optimization model for the traveling salesman problem based on Gavish–Graves (GG) formulation.

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
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
        return m, x

    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        """
        A method to set the objective function

        Args:
            c: cost vector
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector does not match number of cost variables.")
        obj = gp.quicksum(
            c[k] * (self.x[i, j] + self.x[j, i]) for k, (i, j) in enumerate(self.edges)
        )
        self._model.setObjective(obj)

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve model
        """
        self._model.update()
        self._model.optimize()
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for k, (i, j) in enumerate(self.edges):
            if self.x[i, j].x > 1e-2 or self.x[j, i].x > 1e-2:
                sol[k] = 1
        return sol, self._model.objVal

    def _addExtraConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> None:
        """Add a single linear constraint to self._model using the GG variable scheme."""
        self._model.addConstr(
            gp.quicksum(
                coefs[k] * (self.x[i, j] + self.x[j, i]) for k, (i, j) in enumerate(self.edges)
            )
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
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self) -> tuple:
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = gp.Model("tsp")
        # turn off output
        m.Params.outputFlag = 0
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
        return m, x

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._model.update()
        self._model.optimize()
        sol = np.zeros(self.num_cost)
        for k, (i, j) in enumerate(self.edges):
            sol[k] = self.x[i, j].x + self.x[j, i].x
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
    This class is an optimization model for the traveling salesman problem based on Danzig–Fulkerson–Johnson (DFJ) formulation and
    constraint generation.

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self) -> tuple:
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = gp.Model("tsp")
        # turn off output
        m.Params.outputFlag = 0
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
        return m, x

    @staticmethod
    def _subtourelim(model, where):
        """
        A static method to add lazy constraints for subtour elimination
        """
        if where == GRB.Callback.MIPSOL:
            # selected edges
            xvals = model.cbGetSolution(model._x)
            selected = gp.tuplelist((i, j) for i, j in model._x if xvals[i, j] > 1e-2)
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
        obj = gp.quicksum(c[i] * self.x[k] for i, k in enumerate(self.edges))
        self._model.setObjective(obj)

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve model
        """
        self._model.update()
        self._model.optimize(self._subtourelim)
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for i, e in enumerate(self.edges):
            if self.x[e].x > 1e-2:
                sol[i] = 1
        return sol, self._model.objVal

    def _addExtraConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> None:
        """Add a single linear constraint to self._model using the DFJ variable scheme."""
        self._model.addConstr(
            gp.quicksum(coefs[i] * self.x[k] for i, k in enumerate(self.edges)) <= rhs
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


class tspMTZModel(tspABModel):
    """
    This class is an optimization model for the traveling salesman problem based on Miller-Tucker-Zemlin (MTZ) formulation.

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self) -> tuple:
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = gp.Model("tsp")
        # turn off output
        m.Params.outputFlag = 0
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
        return m, x

    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        """
        A method to set the objective function

        Args:
            c: cost vector
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector does not match number of cost variables.")
        obj = gp.quicksum(
            c[k] * (self.x[i, j] + self.x[j, i]) for k, (i, j) in enumerate(self.edges)
        )
        self._model.setObjective(obj)

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve model
        """
        self._model.update()
        self._model.optimize()
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for k, (i, j) in enumerate(self.edges):
            if self.x[i, j].x > 1e-2 or self.x[j, i].x > 1e-2:
                sol[k] = 1
        return sol, self._model.objVal

    def _addExtraConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> None:
        """Add a single linear constraint to self._model using the MTZ variable scheme."""
        self._model.addConstr(
            gp.quicksum(
                coefs[k] * (self.x[i, j] + self.x[j, i]) for k, (i, j) in enumerate(self.edges)
            )
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
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self) -> tuple:
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = gp.Model("tsp")
        # turn off output
        m.Params.outputFlag = 0
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
        return m, x

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._model.update()
        self._model.optimize()
        sol = np.zeros(self.num_cost)
        for k, (i, j) in enumerate(self.edges):
            sol[k] = self.x[i, j].x + self.x[j, i].x
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
