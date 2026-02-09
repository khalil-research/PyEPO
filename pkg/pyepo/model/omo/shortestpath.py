#!/usr/bin/env python
# coding: utf-8
"""
Shortest path problem
"""

from collections import defaultdict

from pyepo.model.omo.omomodel import optOmoModel
from pyepo.model.opt import _get_grid_arcs

try:
    from pyomo import environ as pe
    _HAS_PYOMO = True
except ImportError:
    _HAS_PYOMO = False


class shortestPathModel(optOmoModel):
    """
    This class is an optimization model for the shortest path problem

    Attributes:
        _model (Pyomo model): Pyomo model
        solver (str): optimization solver in the background
        grid (tuple of int): size of grid network
        arcs (list): list of arcs
    """

    def __init__(self, grid, solver="glpk"):
        """
        Args:
            grid (tuple of int): size of grid network
            solver (str): optimization solver in the background
        """
        self.grid = grid
        self.arcs = _get_grid_arcs(grid)
        super().__init__(solver)

    def _getModel(self):
        """
        A method to build Pyomo model
        """
        # create a model
        m = pe.ConcreteModel("shortest path")
        # parameters
        m.arcs = pe.Set(initialize=self.arcs)
        # variables
        x = pe.Var(m.arcs, domain=pe.PositiveReals, bounds=(0,1))
        m.x = x
        # build adjacency lists
        out_arcs = defaultdict(list)
        in_arcs = defaultdict(list)
        for e in self.arcs:
            out_arcs[e[0]].append(e)
            in_arcs[e[1]].append(e)
        # constraints
        m.cons = pe.ConstraintList()
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                expr = sum(x[e] for e in in_arcs[v]) - sum(x[e] for e in out_arcs[v])
                # source
                if i == 0 and j == 0:
                    m.cons.add(expr == -1)
                # sink
                elif i == self.grid[0] - 1 and j == self.grid[1] - 1:
                    m.cons.add(expr == 1)
                # transition
                else:
                    m.cons.add(expr == 0)
        return m, x


if __name__ == "__main__":

    import random
    # random seed
    random.seed(42)
    # set random cost for test
    cost = [random.random() for _ in range(40)]

    # solve model
    optmodel = shortestPathModel(grid=(5,5), solver="gurobi") # init model
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
