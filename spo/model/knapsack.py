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
            weights: weights of items
            capacity: total capacity
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
        assert len(c) == self.num_cost, 'Size of cost vector cannot match vars.'
        obj = gp.quicksum(c[i] * self.x[k] for i, k in enumerate(self.x))
        self._model.setObjective(obj)

    def solve(self):
        """
        solve model
        """
        self._model.update()
        self._model.optimize()
        return [self.x[k].x for k in self.x], self._model.objVal

    def copy(self):
        """
        copy model
        """
        new_model = super().copy()
        # update model
        self._model.update()
        # new model
        new_model._model = self._model.copy()
        # variables for new model
        x = new_model._model.getVars()
        new_model.x = {key: x[i] for i, key in enumerate(self.x)}
        return new_model

    def addConstr(self, coefs, rhs):
        """
        add new constraint
        """
        assert len(coefs) == self.num_cost, 'Size of coef vector cannot cost.'
        # copy
        new_model = self.copy()
        # add constraint
        new_model._model.addConstr(gp.quicksum(coefs[i] * new_model.x[k]
                                               for i, k in enumerate(new_model.x))
                                   <= rhs)
        return new_model

    def relax(self):
        """
        relax model
        """
        # copy
        model_rel = knapsackModelRel(self.weights, self.capacity)
        return model_rel


class knapsackModelRel(knapsackModel):
    """relaxed optimization model for knapsack problem"""

    def _getModel(self):
        """
        Gurobi model
        """
        # ceate a model
        m = gp.Model('knapsack')
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        self.x = m.addVars(self.items, name='x', ub=1)
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstr(gp.quicksum(self.weights[i] * self.x[i] for i in self.items) <= self.capacity)
        return m

    def relax(self):
        """
        relax model
        """
        raise RuntimeError('Model has already been relaxed.')
