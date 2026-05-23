#!/usr/bin/env python
"""
Knapsack problem
"""

from __future__ import annotations

import numpy as np
from coptpy import COPT, Envr

from pyepo.model.copt.coptmodel import optCoptModel


class knapsackModel(optCoptModel):
    """
    This class is an optimization model for the knapsack problem

    Attributes:
        _model (COPT model): COPT model
        weights (np.ndarray): weights of items
        capacity (np.ndarray): total capacity
        items (list): list of item index
    """

    def __init__(self, weights: np.ndarray | list, capacity: np.ndarray | list) -> None:
        """
        Args:
            weights: weights of items
            capacity: total capacity
        """
        self.weights = np.asarray(weights)
        self.capacity = np.asarray(capacity)
        self.items = list(range(self.weights.shape[1]))
        super().__init__()

    def _getModel(self) -> tuple:
        """
        A method to build COPT model
        """
        m = Envr().createModel("knapsack")
        num_items = self.weights.shape[1]
        x = m.addMVar(num_items, vtype=COPT.BINARY, nameprefix="x")
        m.setObjSense(COPT.MAXIMIZE)
        m.addConstr(self.weights @ x <= self.capacity)
        return m, x

    def relax(self) -> knapsackModelRel:
        """
        A method to get linear relaxation model
        """
        # copy
        model_rel = knapsackModelRel(self.weights, self.capacity)
        return model_rel


class knapsackModelRel(knapsackModel):
    """
    This class is relaxed optimization model for knapsack problem.
    """

    def _getModel(self) -> tuple:
        """
        A method to build COPT model
        """
        m = Envr().createModel("knapsack")
        num_items = self.weights.shape[1]
        x = m.addMVar(num_items, lb=0.0, ub=1.0, vtype=COPT.CONTINUOUS, nameprefix="x")
        m.setObjSense(COPT.MAXIMIZE)
        m.addConstr(self.weights @ x <= self.capacity)
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
