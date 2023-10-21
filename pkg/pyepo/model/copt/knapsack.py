#!/usr/bin/env python
# coding: utf-8
"""
Knapsack problem
"""

import numpy as np
from coptpy import Envr
from coptpy import COPT

from pyepo import EPO
from pyepo.model.copt.coptmodel import optCoptModel


class knapsackModel(optCoptModel):
    """
    This class is optimization model for knapsack problem

    Attributes:
        _model (PyOmo model): Pyomo model
        solver (str): optimization solver in the background
        weights (np.ndarray): weights of items
        capacity (np.ndarray): total capacity
        items (list): list of item index
    """

    def __init__(self, weights, capacity):
        """
        Args:
            weights (np.ndarray / list): weights of items
            capacity (np.ndarray / list): total capacity
            solver (str): optimization solver in the background
        """
        self.weights = np.array(weights)
        self.capacity = np.array(capacity)
        self.items = list(range(self.weights.shape[1]))
        super().__init__()

    def _getModel(self):
        """
        A method to build pyomo model
        """
        # ceate a model
        m = Envr().createModel("knapsack")
        # varibles
        x = m.addVars(self.items, vtype=COPT.BINARY)
        # sense
        m.setObjSense(COPT.MAXIMIZE)
        # constraints
        for i in range(len(self.capacity)):
            m.addConstr(sum(self.weights[i,j] * x[j]
                        for j in self.items) <= self.capacity[i])
        return m, x

    def relax(self):
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

    def _getModel(self):
        """
        A method to build pyomo
        """
        # ceate a model
        m = Envr().createModel("knapsack")
        # varibles
        x = m.addVars(self.items, nameprefix='x', vtype=COPT.CONTINUOUS, lb=0, ub=1)
        # sense
        m.setObjSense(COPT.MAXIMIZE)
        # constraints
        for i in range(len(self.capacity)):
            m.addConstr(sum(self.weights[i,j] * x[j]
                        for j in self.items) <= self.capacity[i])
        return m, x

    def relax(self):
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
    weights = np.random.choice(range(300, 800), size=(2,16)) / 100
    capacity = [20, 20]

    # solve model
    optmodel = knapsackModel(weights=weights, capacity=capacity) # init model
    optmodel = optmodel.copy()
    optmodel.setObj(cost) # set objective function
    sol, obj = optmodel.solve() # solve
    # print res
    print('Obj: {}'.format(obj))
    for i in range(16):
        if sol[i] > 1e-3:
            print(i)

    # relax
    optmodel = optmodel.relax()
    optmodel.setObj(cost) # set objective function
    sol, obj = optmodel.solve() # solve
    # print res
    print('Obj: {}'.format(obj))
    for i in range(16):
        if sol[i] > 1e-3:
            print(i)

    # add constraint
    optmodel = optmodel.addConstr([weights[0,i] for i in range(16)], 10)
    optmodel.setObj(cost) # set objective function
    sol, obj = optmodel.solve() # solve
    # print res
    print('Obj: {}'.format(obj))
    for i in range(16):
        if sol[i] > 1e-3:
            print(i)
