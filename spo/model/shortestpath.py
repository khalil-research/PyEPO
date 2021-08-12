#!/usr/bin/env python
# coding: utf-8

import gurobipy as gp
from gurobipy import GRB
from spo.model import optModel

class shortestPathModel(optModel):
    """
    This class is optimization model for shortest path problem

    Args:
        grid: size of grid network
    """

    def __init__(self, grid):
        self.grid = grid
        self.arcs = self._getArcs()
        super().__init__()

    def _getArcs(self):
        """
        A method to get list of arcs for grid network

        Returns:
            list: arcs
        """
        arcs = []
        for i in range(self.grid[0]):
            # edges on rows
            for j in range(self.grid[1]-1):
                v = i * self.grid[1] + j
                arcs.append((v,v+1))
            # edges in columns
            if i == self.grid[0] - 1:
                continue
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                arcs.append((v,v+self.grid[1]))
        return arcs

    @property
    def num_cost(self):
        return len(self.arcs)

    def _getModel(self):
        """
        A method to build Gurobi model
        """
        # ceate a model
        m = gp.Model('shortest path')
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        self.x = m.addVars(self.arcs, name='x')
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                v = v = i * self.grid[1] + j
                expr = 0
                for e in self.arcs:
                    # flow in
                    if v == e[1]:
                        expr += self.x[e]
                    # flow out
                    elif v == e[0]:
                        expr -= self.x[e]
                # source
                if i == 0 and j == 0:
                    m.addConstr(expr == -1)
                # sink
                elif i == self.grid[0] - 1 and j == self.grid[0] - 1:
                    m.addConstr(expr == 1)
                # transition
                else:
                    m.addConstr(expr == 0)
        return m

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (ndarray): cost of objective function
        """
        assert len(c) == self.num_cost, 'Size of cost vector cannot match vars.'
        obj = gp.quicksum(c[i] * self.x[k] for i, k in enumerate(self.x))
        self._model.setObjective(obj)

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._model.update()
        self._model.optimize()
        return [self.x[k].x for k in self.x], self._model.objVal

    def copy(self):
        """
        A method to copy model

        Returns:
            optModel: new copied model
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
        A method to add new constraint

        Args:
            coefs (ndarray): coeffcients of new constraint
            rhs (float): right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        assert len(coefs) == self.num_cost, 'Size of coef vector cannot cost.'
        # copy
        new_model = self.copy()
        # add constraint
        new_model._model.addConstr(gp.quicksum(coefs[i] * new_model.x[k]
                                               for i, k in enumerate(new_model.x))
                                   <= rhs)
        return new_model
