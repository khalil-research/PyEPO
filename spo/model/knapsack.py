#!/usr/bin/env python
# coding: utf-8

import gurobipy as gp
from gurobipy import GRB
from spo.model import optModel

class knapsackModel(optModel):
    """optimization model for knapsack problem"""

    def __init__(self, weights, capacity):
        """
        Args:
            grid: size of grid network
        """
        self.weights = weights
        self.capacity = capacity
        self.items = [i for i in range(len(self.weights))]
        super().__init__()

    @property
    def num_cost(self):
        return len(self.items)

    def _getModel(self):
        """
        Gurobi model
        """
        # ceate a model
        m = gp.Model('knapsack')
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        self.x = m.addVars(self.items, name='x', vtype=GRB.BINARY)
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstr(gp.quicksum(self.weights[i] * self.x[i] for i in self.items) <= self.capacity)
        return m

    def setObj(self, c):
        """
        set objective function
        """
        assert len(c) == len(self.items), 'Size of cost vector cannot match items'
        obj = gp.quicksum(c[i] * self.x[i] for i in self.items)
        self._model.setObjective(obj)

    def solve(self):
        """
        solve model
        """
        self._model.update()
        self._model.optimize()
        return [self.x[i].x for i in self.items], self._model.objVal

    def addConstr(self, coefs, rhs):
        """
        add new constraint
        """
        assert len(coefs) == len(self.items), 'Size of coef vector cannot match items'
        # copy
        new_model = knapsackModel(self.weights, self.capacity)
        # add constraint
        new_model._model.addConstr(gp.quicksum(coefs[i] * new_model.x[i]
                                               for i in range(self.num_cost))
                                   <= rhs)
        return new_model

    def clean(self):
        """
        clean model
        Returns:
            None
        """
        self._model.dispose()
