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

from pyepo.model.bases import shortestPathBase
from pyepo.model.grb.grbmodel import optGrbModel
from pyepo.model.utils import _incidence_matrix


class shortestPathModel(shortestPathBase, optGrbModel):
    """
    Gurobi-backed shortest path on a grid network.

    Attributes:
        _model (GurobiPy model): Gurobi model
        grid (tuple of int): Size of grid network
        arcs (list): List of arcs
    """

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
        # sparse node-arc incidence: row v sums (in-flow) - (out-flow) at v
        A = _incidence_matrix(self.arcs, num_nodes)
        b = np.zeros(num_nodes, dtype=np.float64)
        b[0] = -1.0
        b[num_nodes - 1] = 1.0
        m.addConstr(A @ x == b)  # type: ignore[call-overload, arg-type]
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
