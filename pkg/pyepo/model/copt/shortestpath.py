#!/usr/bin/env python
# coding: utf-8
"""
Shortest path problem
"""

from collections import defaultdict

from coptpy import Envr
from coptpy import COPT

from pyepo.model.copt.coptmodel import optCoptModel
from pyepo.model.opt import _get_grid_arcs


class shortestPathModel(optCoptModel):
    """
    This class is an optimization model for the shortest path problem

    Attributes:
        _model (COPT model): COPT model
        grid (tuple of int): size of grid network
        arcs (list): list of arcs
    """

    def __init__(self, grid):
        """
        Args:
            grid (tuple of int): size of grid network
        """
        self.grid = grid
        self.arcs = _get_grid_arcs(grid)
        super().__init__()

    def _getModel(self):
        """
        A method to build COPT model
        """
        # create a model
        m = Envr().createModel("shortest path")
        # variables
        x = m.addVars(self.arcs, nameprefix='x', vtype=COPT.CONTINUOUS, lb=0, ub=1)
        # sense
        m.setObjSense(COPT.MINIMIZE)
        # build adjacency lists
        out_arcs = defaultdict(list)
        in_arcs = defaultdict(list)
        for e in self.arcs:
            out_arcs[e[0]].append(e)
            in_arcs[e[1]].append(e)
        # constraints
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                expr = sum(x[e] for e in in_arcs[v]) - sum(x[e] for e in out_arcs[v])
                # source
                if i == 0 and j == 0:
                    m.addConstr(expr == -1)
                # sink
                elif i == self.grid[0] - 1 and j == self.grid[1] - 1:
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
