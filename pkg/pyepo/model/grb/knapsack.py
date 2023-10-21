#!/usr/bin/env python
# coding: utf-8
"""
Knapsack problem
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from pyepo.model.grb.grbmodel import optGrbModel


class knapsackModel(optGrbModel):
    """
    This class is optimization model for knapsack problem

    Attributes:
        _model (GurobiPy model): Gurobi model
        weights (np.ndarray / list): Weights of items
        capacity (np.ndarray / listy): Total capacity
        items (list): List of item index
    """

    def __init__(self, weights, capacity):
        """
        Args:
            weights (np.ndarray / list): weights of items
            capacity (np.ndarray / list): total capacity
        """
        self.weights = np.array(weights)
        self.capacity = np.array(capacity)
        self.items = list(range(self.weights.shape[1]))
        super().__init__()

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("knapsack")
        # varibles
        x = m.addVars(self.items, name="x", vtype=GRB.BINARY)
        # sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        for i in range(len(self.capacity)):
            m.addConstr(gp.quicksum(self.weights[i,j] * x[j]
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
        A method to build Gurobi
        """
        # ceate a model
        m = gp.Model("knapsack")
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        x = m.addVars(self.items, name="x", ub=1)
        # sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        for i in range(len(self.capacity)):
            m.addConstr(gp.quicksum(self.weights[i,j] * x[j]
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
