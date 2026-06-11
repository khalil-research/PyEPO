#!/usr/bin/env python
"""
Knapsack problem
"""

from __future__ import annotations

import numpy as np

from pyepo.model.bases import knapsackBase
from pyepo.model.omo.omomodel import optOmoModel

try:
    from pyomo import environ as pe
except ImportError:
    pass


class knapsackModel(knapsackBase, optOmoModel):
    """
    Pyomo-backed knapsack.

    Attributes:
        _model (Pyomo model): Pyomo model
        solver (str): optimization solver in the background
        weights (np.ndarray): weights of items
        capacity (np.ndarray): total capacity
        items (list): list of item index
    """

    def __init__(
        self,
        weights: np.ndarray | list,
        capacity: np.ndarray | list,
        solver: str = "glpk",
    ) -> None:
        """
        Args:
            weights: weights of items
            capacity: total capacity
            solver: optimization solver in the background
        """
        super().__init__(weights, capacity, solver)

    def _getModel(self) -> tuple:
        """
        A method to build Pyomo model
        """
        # create a model
        m = pe.ConcreteModel("knapsack")
        # parameters
        m.its = pe.Set(initialize=self.items)
        # variables
        x = pe.Var(m.its, domain=pe.Binary)
        m.x = x
        # constraints
        m.cons = pe.ConstraintList()
        for i in range(len(self.capacity)):
            m.cons.add(sum(self.weights[i, j] * x[j] for j in self.items) <= self.capacity[i])
        return m, x

    def relax(self) -> knapsackModelRel:
        """
        A method to get linear relaxation model
        """
        model_rel = knapsackModelRel(self.weights, self.capacity, self.solver)
        # replay user cuts on the relaxation
        for coefs, rhs in self._extra_constrs:
            model_rel = model_rel.addConstr(coefs, rhs)
        return model_rel


class knapsackModelRel(knapsackModel):
    """
    LP relaxation of the Pyomo knapsack.
    """

    def _getModel(self) -> tuple:
        """
        A method to build Pyomo model
        """
        # create a model
        m = pe.ConcreteModel("knapsack")
        # parameters
        m.its = pe.Set(initialize=self.items)
        # variables
        x = pe.Var(m.its, domain=pe.NonNegativeReals, bounds=(0, 1))
        m.x = x
        # constraints
        m.cons = pe.ConstraintList()
        for i in range(len(self.capacity)):
            m.cons.add(sum(self.weights[i, j] * x[j] for j in self.items) <= self.capacity[i])
        return m, x

    def relax(self) -> knapsackModelRel:
        """
        A forbidden method to relax MIP model
        """
        raise RuntimeError("Model has already been relaxed.")


if __name__ == "__main__":
    import random

    # random seed
    random.seed(42)
    # set random cost for test
    cost = [random.random() for _ in range(16)]
    weights = np.random.choice(range(300, 800), size=(2, 16)) / 100
    capacity = [20, 20]

    # solve model
    optmodel = knapsackModel(weights=weights, capacity=capacity, solver="gurobi")  # init model
    optmodel = optmodel.copy()
    optmodel.setObj(cost)  # set objective function
    sol, obj = optmodel.solve()  # solve
    # print res
    print(f"Obj: {obj}")
    for i in range(16):
        if sol[i] > 1e-3:
            print(i)

    # relax
    optmodel = optmodel.relax()
    optmodel.setObj(cost)  # set objective function
    sol, obj = optmodel.solve()  # solve
    # print res
    print(f"Obj: {obj}")
    for i in range(16):
        if sol[i] > 1e-3:
            print(i)

    # add constraint
    optmodel = optmodel.addConstr([weights[0, i] for i in range(16)], 10)
    optmodel.setObj(cost)  # set objective function
    sol, obj = optmodel.solve()  # solve
    # print res
    print(f"Obj: {obj}")
    for i in range(16):
        if sol[i] > 1e-3:
            print(i)
