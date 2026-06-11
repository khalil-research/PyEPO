#!/usr/bin/env python
"""
Knapsack problem
"""

from __future__ import annotations

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    pass

from pyepo.model.bases import knapsackBase
from pyepo.model.grb.grbmodel import optGrbModel


class knapsackModel(knapsackBase, optGrbModel):
    """
    Gurobi-backed knapsack.

    Attributes:
        _model (GurobiPy model): Gurobi model
        weights (np.ndarray): Weights of items
        capacity (np.ndarray): Total capacity
        items (list): List of item index
    """

    def _getModel(self) -> tuple:
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = gp.Model("knapsack")
        # variables
        x = m.addMVar(len(self.items), name="x", vtype=GRB.BINARY)
        # sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(self.weights @ x <= self.capacity)
        return m, x

    def relax(self) -> knapsackModelRel:
        """
        A method to get linear relaxation model
        """
        # copy
        model_rel = knapsackModelRel(self.weights, self.capacity)
        # replay user cuts on the relaxation
        for coefs, rhs in self._extra_constrs:
            model_rel = model_rel.addConstr(coefs, rhs)
        return model_rel


class knapsackModelRel(knapsackModel):
    """
    LP relaxation of the Gurobi knapsack.
    """

    def _getModel(self) -> tuple:
        """
        A method to build Gurobi model
        """
        # create a model
        m = gp.Model("knapsack")
        # variables
        x = m.addMVar(len(self.items), name="x", ub=1)
        # sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(self.weights @ x <= self.capacity)
        return m, x

    def relax(self) -> knapsackModelRel:
        """
        A forbidden method to relax MIP model
        """
        raise RuntimeError("Model has already been relaxed.")


if __name__ == "__main__":
    # random seed
    np.random.seed(42)
    # set random cost for test
    cost = np.random.random(16)
    weights = np.random.choice(range(300, 800), size=(2, 16)) / 100
    capacity = [20, 20]

    # solve model
    optmodel = knapsackModel(weights=weights, capacity=capacity)  # init model
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
