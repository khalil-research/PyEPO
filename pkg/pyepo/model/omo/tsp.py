#!/usr/bin/env python
"""
Traveling salesman problem
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyepo.model.omo.omomodel import optOmoModel
from pyepo.model.utils import getTspTour
from pyepo.utils import costToNumpy

if TYPE_CHECKING:
    import torch

try:
    from pyomo import environ as pe
except ImportError:
    pass


class tspABModel(optOmoModel):
    """
    This abstract class is an optimization model for the traveling salesman problem.
    This model is for further implementation of different formulation.

    Attributes:
        _model (Pyomo model): Pyomo model
        solver (str): optimization solver in the background
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def __init__(self, num_nodes: int, solver: str = "glpk") -> None:
        """
        Args:
            num_nodes: number of nodes
            solver: optimization solver in the background
        """
        self.num_nodes = num_nodes
        self.nodes = list(range(num_nodes))
        self.edges = [(i, j) for i in self.nodes for j in self.nodes if i < j]
        super().__init__(solver)
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
        new_model = type(self)(self.num_nodes, self.solver)
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
        _model (Pyomo model): Pyomo model
        solver (str): optimization solver in the background
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self) -> tuple:
        """
        A method to build Pyomo model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = pe.ConcreteModel("tsp")
        # parameters
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        m.dedges = pe.Set(initialize=directed_edges)
        # variables
        x = pe.Var(m.dedges, domain=pe.Binary)
        m.x = x
        y = pe.Var(m.dedges, domain=pe.NonNegativeReals)
        m.y = y
        # constraints
        m.cons = pe.ConstraintList()
        for j in self.nodes:
            m.cons.add(sum(x[i, j] for i in self.nodes if i != j) == 1)
        for i in self.nodes:
            m.cons.add(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for i in self.nodes[1:]:
            m.cons.add(
                sum(y[i, j] for j in self.nodes if j != i)
                - sum(y[j, i] for j in self.nodes[1:] if j != i)
                == 1
            )
        for i, j in directed_edges:
            if i != 0:
                m.cons.add(y[i, j] <= (len(self.nodes) - 1) * x[i, j])
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
        # delete previous component
        self._model.del_component(self._model.obj)
        # set obj
        obj = sum(c[k] * (self.x[i, j] + self.x[j, i]) for k, (i, j) in enumerate(self.edges))
        self._model.obj = pe.Objective(sense=pe.minimize, expr=obj)

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve model
        """
        self._solverfac.solve(self._model)
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for k, (i, j) in enumerate(self.edges):
            if pe.value(self.x[i, j]) > 1e-2 or pe.value(self.x[j, i]) > 1e-2:
                sol[k] = 1
        return sol, pe.value(self._model.obj)

    def _addExtraConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> None:
        """Add a single linear constraint to self._model using the GG variable scheme."""
        self._model.cons.add(
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
        model_rel = tspGGModelRel(self.num_nodes, self.solver)
        self._replay_extras(model_rel)
        return model_rel


class tspGGModelRel(tspGGModel):
    """
    This class is relaxation of tspGGModel.

    Attributes:
        _model (Pyomo model): Pyomo model
        solver (str): optimization solver in the background
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self) -> tuple:
        """
        A method to build Pyomo model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = pe.ConcreteModel("tsp")
        # parameters
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        m.dedges = pe.Set(initialize=directed_edges)
        # variables
        x = pe.Var(m.dedges, domain=pe.NonNegativeReals, bounds=(0, 1))
        m.x = x
        y = pe.Var(m.dedges, domain=pe.NonNegativeReals)
        m.y = y
        # constraints
        m.cons = pe.ConstraintList()
        for j in self.nodes:
            m.cons.add(sum(x[i, j] for i in self.nodes if i != j) == 1)
        for i in self.nodes:
            m.cons.add(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for i in self.nodes[1:]:
            m.cons.add(
                sum(y[i, j] for j in self.nodes if j != i)
                - sum(y[j, i] for j in self.nodes[1:] if j != i)
                == 1
            )
        for i, j in directed_edges:
            if i != 0:
                m.cons.add(y[i, j] <= (len(self.nodes) - 1) * x[i, j])
        return m, x

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._solverfac.solve(self._model)
        sol = np.zeros(self.num_cost)
        for k, (i, j) in enumerate(self.edges):
            sol[k] = pe.value(self.x[i, j]) + pe.value(self.x[j, i])
        return sol, pe.value(self._model.obj)

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


class tspMTZModel(tspABModel):
    """
    This class is an optimization model for the traveling salesman problem based on Miller-Tucker-Zemlin (MTZ) formulation.

    Attributes:
        _model (Pyomo model): Pyomo model
        solver (str): optimization solver in the background
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self) -> tuple:
        """
        A method to build Pyomo model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = pe.ConcreteModel("tsp")
        # parameters
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        m.dedges = pe.Set(initialize=directed_edges)
        m.nds = pe.Set(initialize=self.nodes)
        # variables
        x = pe.Var(m.dedges, domain=pe.Binary)
        m.x = x
        u = pe.Var(m.nds, domain=pe.NonNegativeReals)
        m.u = u
        # constraints
        m.cons = pe.ConstraintList()
        for j in self.nodes:
            m.cons.add(sum(x[i, j] for i in self.nodes if i != j) == 1)
        for i in self.nodes:
            m.cons.add(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for i, j in directed_edges:
            if (i != 0) and (j != 0):
                m.cons.add(u[j] - u[i] >= len(self.nodes) * (x[i, j] - 1) + 1)
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
        # delete previous component
        self._model.del_component(self._model.obj)
        # set obj
        obj = sum(c[k] * (self.x[i, j] + self.x[j, i]) for k, (i, j) in enumerate(self.edges))
        self._model.obj = pe.Objective(sense=pe.minimize, expr=obj)

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve model
        """
        self._solverfac.solve(self._model)
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for k, (i, j) in enumerate(self.edges):
            if pe.value(self.x[i, j]) > 1e-2 or pe.value(self.x[j, i]) > 1e-2:
                sol[k] = 1
        return sol, pe.value(self._model.obj)

    def _addExtraConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> None:
        """Add a single linear constraint to self._model using the MTZ variable scheme."""
        self._model.cons.add(
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
        model_rel = tspMTZModelRel(self.num_nodes, self.solver)
        self._replay_extras(model_rel)
        return model_rel


class tspMTZModelRel(tspMTZModel):
    """
    This class is relaxation of tspMTZModel.

    Attributes:
        _model (Pyomo model): Pyomo model
        solver (str): optimization solver in the background
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self) -> tuple:
        """
        A method to build Pyomo model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = pe.ConcreteModel("tsp")
        # parameters
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        m.dedges = pe.Set(initialize=directed_edges)
        m.nds = pe.Set(initialize=self.nodes)
        # variables
        x = pe.Var(m.dedges, domain=pe.NonNegativeReals, bounds=(0, 1))
        m.x = x
        u = pe.Var(m.nds, domain=pe.NonNegativeReals)
        m.u = u
        # constraints
        m.cons = pe.ConstraintList()
        for j in self.nodes:
            m.cons.add(sum(x[i, j] for i in self.nodes if i != j) == 1)
        for i in self.nodes:
            m.cons.add(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for i, j in directed_edges:
            if (i != 0) and (j != 0):
                m.cons.add(u[j] - u[i] >= len(self.nodes) * (x[i, j] - 1) + 1)
        return m, x

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._solverfac.solve(self._model)
        sol = np.zeros(self.num_cost)
        for k, (i, j) in enumerate(self.edges):
            sol[k] = pe.value(self.x[i, j]) + pe.value(self.x[j, i])
        return sol, pe.value(self._model.obj)

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
    optmodel = tspGGModel(num_nodes=num_nodes, solver="gurobi")
    optmodel = optmodel.copy()
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print(f"GG Obj: {obj}")
    tour = optmodel.getTour(sol)
    print(f"GG Tour: {tour}")

    # solve MTZ model
    optmodel = tspMTZModel(num_nodes=num_nodes, solver="gurobi")
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print(f"MTZ Obj: {obj}")
    tour = optmodel.getTour(sol)
    print(f"MTZ Tour: {tour}")

    # relax GG model
    optmodel = tspGGModel(num_nodes=num_nodes, solver="gurobi")
    optmodel = optmodel.relax()
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print(f"GG Relaxed Obj: {obj}")

    # add constraint
    optmodel = tspMTZModel(num_nodes=num_nodes, solver="gurobi")
    optmodel = optmodel.addConstr([1] * num_edges, num_edges - 1)
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print(f"MTZ + Constr Obj: {obj}")
