#!/usr/bin/env python
# coding: utf-8
"""
Shortest path problem
"""

from coptpy import Envr
from coptpy import COPT

from pyepo.model.copt.coptmodel import optCoptModel


class shortestPathModel(optCoptModel):
    """
    This class is optimization model for shortest path problem

    Attributes:
        _model (PyOmo model): Pyomo model
        solver (str): optimization solver in the background
        grid (tuple of int): size of grid network
        arcs (list): list of arcs
    """

    def __init__(self, grid):
        """
        Args:
            grid (tuple of int): size of grid network
            solver (str): optimization solver in the background
        """
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
            for j in range(self.grid[1] - 1):
                v = i * self.grid[1] + j
                arcs.append((v, v + 1))
            # edges in columns
            if i == self.grid[0] - 1:
                continue
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                arcs.append((v, v + self.grid[1]))
        return arcs

    def _getModel(self):
        """
        A method to build pyomo model
        """
        # ceate a model
        m = Envr().createModel("shortest path")
        # varibles
        x = m.addVars(self.arcs, nameprefix='x', vtype=COPT.CONTINUOUS, lb=0, ub=1)
        # sense
        m.setObjSense(COPT.MINIMIZE)
        # constraints
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                v = v = i * self.grid[1] + j
                expr = 0
                for e in self.arcs:
                    # flow in
                    if v == e[1]:
                        expr += x[e]
                    # flow out
                    elif v == e[0]:
                        expr -= x[e]
                # source
                if i == 0 and j == 0:
                    m.addConstr(expr == -1)
                # sink
                elif i == self.grid[0] - 1 and j == self.grid[0] - 1:
                    m.addConstr(expr == 1)
                # transition
                else:
                    m.addConstr(expr == 0)
        return m, x


if __name__ == "__main__":

    import random
    # random seed
    random.seed(42)
    # set random cost for test
    cost = [random.random() for _ in range(40)]

    # solve model
    optmodel = shortestPathModel(grid=(5,5)) # init model
    optmodel = optmodel.copy()
    optmodel.setObj(cost) # set objective function
    sol, obj = optmodel.solve() # solve
    # print res
    print('Obj: {}'.format(obj))
    for i, e in enumerate(optmodel.arcs):
        if sol[i] > 1e-3:
            print(e)


    # add constraint
    optmodel = optmodel.addConstr([1]*40, 30)
    optmodel.setObj(cost) # set objective function
    sol, obj = optmodel.solve() # solve
    # print res
    print('Obj: {}'.format(obj))
    for i, e in enumerate(optmodel.arcs):
        if sol[i] > 1e-3:
            print(e)
