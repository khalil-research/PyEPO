#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
from spo.model import optModel

class tspModel(optModel):
    """optimization model for traveling salesman problem"""

    def __init__(self, num_nodes):
        """
        Args:
            num_nodes: number of nodes
        """
        self.nodes = [i for i in range(num_nodes)]
        self.edges = [(i,j) for i in range(num_nodes) for j in range(num_nodes) if i < j]
        super().__init__()

    @property
    def num_cost(self):
        return len(self.edges)

    def _getModel(self):
        """
        Gurobi model for DFJ formulation
        """
        # ceate a model
        m = gp.Model('tsp')
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        self.x = m.addVars(self.edges, name='x', vtype=GRB.BINARY)
        for i, j in self.edges:
            self.x[j, i] = self.x[i, j]
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstrs(self.x.sum(i, '*') == 2 for i in self.nodes) ## 2 degree
        # activate lazy constraints
        m._x = self.x
        m._n = len(self.nodes)
        m.Params.lazyConstraints = 1
        return m

    @staticmethod
    def subtourelim(model, where):
        """
        lazy constraints for subtour elimination
        """

        def subtour(selected, n):
            """
            find shortest cycle
            """
            unvisited = list(range(n))
            # init dummy longest cycle
            cycle = range(n+1)
            while unvisited:
                thiscycle = []
                neighbors = unvisited
                while neighbors:
                    current = neighbors[0]
                    thiscycle.append(current)
                    unvisited.remove(current)
                    neighbors = [j for i, j in selected.select(current, '*') if j in unvisited]
                if len(cycle) > len(thiscycle):
                    cycle = thiscycle
            return cycle

        if where == GRB.Callback.MIPSOL:
            # selected edges
            xvals = model.cbGetSolution(model._x)
            selected = gp.tuplelist((i, j) for i, j in model._x.keys() if xvals[i, j] > 1e-2)
            # shortest cycle
            tour = subtour(selected, model._n)
            # add cuts
            if len(tour) < model._n:
                model.cbLazy(gp.quicksum(model._x[i, j]
                                         for i, j in combinations(tour, 2))
                             <= len(tour)-1)


    def setObj(self, c):
        """
        set objective function
        """
        assert len(c) == len(self.edges), 'Size of cost vector cannot match edges.'
        obj = gp.quicksum(c[i] * self.x[e] for i, e in enumerate(self.edges))
        self._model.setObjective(obj)

    def solve(self):
        """
        solve model
        """
        self._model.update()
        self._model.optimize(self.subtourelim)
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for i, e in enumerate(self.edges):
            if self.x[e].x > 1e-2:
                sol[i] = 1
        return sol, self._model.objVal

    def getTour(self, sol):
        """
        get a tour from solution
        """
        # active edges
        edges = defaultdict(list)
        for i, (j,k) in enumerate(self.edges):
            if sol[i] > 1e-2:
                edges[j].append(k)
                edges[k].append(j)
        # get tour
        visited = {0}
        tour = [0]
        while len(visited) < len(edges):
            i = tour[-1]
            for j in edges[i]:
                if j not in visited:
                    tour.append(j)
                    visited.add(j)
                    break
        if 0 in edges[tour[-1]]:
            tour.append(0)
        return tour

    def addConstr(self, coefs, rhs):
        """
        add new constraint
        """
        assert len(coefs) == len(self.edges), 'Size of coef vector cannot match edges.'
        # copy
        new_model = tspModel(len(self.nodes))
        # add constraint
        new_model._model.addConstr(gp.quicksum(coefs[i] * new_model.x[e]
                                               for i, e in enumerate(self.edges))
                                   <= rhs)
        return new_model
