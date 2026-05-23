#!/usr/bin/env python
"""
Shortest path problem
"""

from __future__ import annotations

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    pass

from pyepo.model.grb.grbmodel import optGrbModel
from pyepo.model.utils import _get_grid_arcs


class shortestPathModel(optGrbModel):
    """
    This class is an optimization model for the shortest path problem

    Attributes:
        _model (GurobiPy model): Gurobi model
        grid (tuple of int): Size of grid network
        arcs (list): List of arcs
    """

    def __init__(self, grid: tuple[int, int]) -> None:
        """
        Args:
            grid: size of grid network
        """
        self.grid = grid
        self.arcs = _get_grid_arcs(grid)
        super().__init__()

    def _getModel(self) -> tuple:
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        m = gp.Model("shortest path")
        num_nodes = self.grid[0] * self.grid[1]
        num_arcs = len(self.arcs)
        x = m.addMVar(num_arcs, ub=1.0, name="x")
        m.modelSense = GRB.MINIMIZE
        # node-arc incidence: row v sums (in-flow) - (out-flow) at v
        A = np.zeros((num_nodes, num_arcs), dtype=np.float64)
        for a, (u, v) in enumerate(self.arcs):
            A[u, a] = -1.0
            A[v, a] = +1.0
        b = np.zeros(num_nodes, dtype=np.float64)
        b[0] = -1.0
        b[num_nodes - 1] = 1.0
        m.addConstr(A @ x == b)
        return m, x


if __name__ == "__main__":
    import random

    # random seed
    random.seed(42)
    # set random cost for test
    cost = [random.random() for _ in range(40)]

    # solve model
    optmodel = shortestPathModel(grid=(5, 5))  # init model
    optmodel = optmodel.copy()
    optmodel.setObj(cost)  # set objective function
    sol, obj = optmodel.solve()  # solve
    # print res
    print(f"Obj: {obj}")
    for i, e in enumerate(optmodel.arcs):
        if sol[i] > 1e-3:
            print(e)

    # add constraint
    optmodel = optmodel.addConstr([1] * 40, 30)
    optmodel.setObj(cost)  # set objective function
    sol, obj = optmodel.solve()  # solve
    # print res
    print(f"Obj: {obj}")
    for i, e in enumerate(optmodel.arcs):
        if sol[i] > 1e-3:
            print(e)
