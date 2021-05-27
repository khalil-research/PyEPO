#!/usr/bin/env python
# coding: utf-8

import gurobipy as gp
from gurobipy import GRB
from spo.model import optModel

class shortestPathModel(optModel):
    """optimization model for shortest path problem"""

    def __init__(self, grid):
        """
        Args:
            grid: size of grid network
        """
        self.grid = grid
        self.arcs = self._getArcs()
        super().__init__()

    def _getArcs(self):
        """
        get list of arcs for grid network
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
        Gurobi model
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
        set objective function
        """
        assert len(c) == len(self.arcs), 'Size of cost vector cannot match arcs'
        obj = gp.quicksum(c[i] * self.x[self.arcs[i]] for i in range(len(self.arcs)))
        self._model.setObjective(obj)

    def solve(self):
        """
        solve model
        """
        self._model.update()
        self._model.optimize()
        return [self.x[e].x for e in self.arcs], self._model.objVal

    def addConstr(self, coefs, rhs):
        """
        add new constraint
        """
        assert len(coefs) == len(self.arcs), 'Size of coef vector cannot match arcs'
        # copy
        new_model = shortestPathModel(self.grid)
        # add constraint
        new_model._model.addConstr(gp.quicksum(coefs[i] * new_model.x[self.arcs[i]]
                                               for i in range(len(self.arcs)))
                                   == rhs)
        return new_model
