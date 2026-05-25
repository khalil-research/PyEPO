#!/usr/bin/env python
"""
Traveling salesman problem
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import numpy as np

from pyepo.model.bases import tspABBase
from pyepo.model.omo.omomodel import optOmoModel
from pyepo.model.utils import _EDGE_ACTIVE_TOL

if TYPE_CHECKING:
    import torch

try:
    from pyomo import environ as pe
except ImportError:
    pass


class tspABModel(tspABBase, optOmoModel):
    """
    Pyomo-backed TSP abstract base. Provides paired-variable objective /
    ``solve`` / ``_addExtraConstr`` shared by GG and MTZ. Pyomo lacks easy
    callback support, so no DFJ formulation exists for this backend.
    """

    def __init__(self, num_nodes: int, solver: str = "glpk") -> None:
        """
        Args:
            num_nodes: number of nodes
            solver: optimization solver in the background
        """
        super().__init__(num_nodes, solver)

    def _new_instance(self) -> tspABModel:
        """Override: omo carries a ``solver`` ctor arg."""
        return type(self)(self.num_nodes, self.solver)

    def _obj_expr(self):
        """Paired: each undirected edge cost weights x[i,j] + x[j,i]."""
        return sum(
            self._model.cost[k] * (self.x[i, j] + self.x[j, i])
            for k, (i, j) in enumerate(self.edges)
        )

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve model
        """
        self._solverfac.solve(self._model)
        sol = np.zeros(self.num_cost, dtype=np.float32)
        for k, (i, j) in enumerate(self.edges):
            if (
                pe.value(self.x[i, j]) > _EDGE_ACTIVE_TOL
                or pe.value(self.x[j, i]) > _EDGE_ACTIVE_TOL
            ):
                sol[k] = 1
        return sol, float(pe.value(self._model.obj))

    def _addExtraConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> None:
        """Add a single linear constraint to ``self._model`` using paired (x[i,j] + x[j,i])."""
        self._model.cons.add(
            sum(coefs[k] * (self.x[i, j] + self.x[j, i]) for k, (i, j) in enumerate(self.edges))
            <= rhs
        )


class tspGGModel(tspABModel):
    """
    Gavish-Graves (GG) formulation.
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

    def relax(self) -> tspGGModelRel:
        """
        A method to get linear relaxation model
        """
        model_rel = tspGGModelRel(self.num_nodes, self.solver)
        self._replay_extras(model_rel)
        return model_rel


class tspGGModelRel(tspGGModel):
    """
    LP relaxation of the GG formulation.
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
        A method to solve model — returns fractional solution.
        """
        self._solverfac.solve(self._model)
        sol = np.zeros(self.num_cost, dtype=np.float32)
        for k, (i, j) in enumerate(self.edges):
            sol[k] = float(pe.value(self.x[i, j])) + float(pe.value(self.x[j, i]))
        return sol, float(pe.value(self._model.obj))

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


class tspMTZModel(tspABModel):
    """
    Miller-Tucker-Zemlin (MTZ) formulation.
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

    def relax(self) -> tspMTZModelRel:
        """
        A method to get linear relaxation model
        """
        model_rel = tspMTZModelRel(self.num_nodes, self.solver)
        self._replay_extras(model_rel)
        return model_rel


class tspMTZModelRel(tspMTZModel):
    """
    LP relaxation of the MTZ formulation.
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
        A method to solve model — returns fractional solution.
        """
        self._solverfac.solve(self._model)
        sol = np.zeros(self.num_cost, dtype=np.float32)
        for k, (i, j) in enumerate(self.edges):
            sol[k] = float(pe.value(self.x[i, j])) + float(pe.value(self.x[j, i]))
        return sol, float(pe.value(self._model.obj))

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
