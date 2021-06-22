#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
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
        assert len(c.shape) == 2, 'Dimension of distance matrix should be 2.'
        assert c.shape[0] == len(self.nodes), 'Row of distance matrix cannot match items.'
        assert c.shape[1] == len(self.nodes), 'Column of distance matrix cannot match items.'
        obj = gp.quicksum(c[i,j] * self.x[i,j] for i, j in self.edges)
        self._model.setObjective(obj)

    def solve(self):
        """
        solve model
        """
        self._model.update()
        self._model.optimize(self.subtourelim)
        return {(i, j): self.x[i,j].x for i, j in self.edges}, self._model.objVal

    @staticmethod
    def getTour(sol):
        """
        get a tour from solution
        """
        # active edges
        edges = defaultdict(list)
        for i, j in sol:
            if sol[i,j] > 1e-2:
                edges[i].append(j)
                edges[j].append(i)
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
        assert len(coefs.shape) == 2, 'Dimension of distance matrix should be 2.'
        assert coefs.shape[0] == len(self.nodes), 'Row of distance matrix cannot match items.'
        assert coefs.shape[1] == len(self.nodes), 'Column of distance matrix cannot match items.'
        # copy
        new_model = tspModel(len(self.nodes))
        # add constraint
        new_model._model.addConstr(gp.quicksum(coefs[i,j] * new_model.x[i,j]
                                               for i, j in self.edges)
                                   <= rhs)
        return new_model
