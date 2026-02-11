#!/usr/bin/env python
# coding: utf-8
"""
Shortest path problem
"""

import numpy as np

try:
    import jax.numpy as jnp
    _HAS_MPAX = True
except ImportError:
    _HAS_MPAX = False

from pyepo.model.mpax.mpaxmodel import optMpaxModel
from pyepo.model.opt import _get_grid_arcs


class shortestPathModel(optMpaxModel):
    """
    This class is an optimization model for the shortest path problem

    Attributes:
        grid (tuple of int): Size of grid network
        arcs (list): List of arcs
    """

    def __init__(self, grid):
        """
        Args:
            grid (tuple of int): size of grid network
        """
        self.grid = grid
        self.arcs = _get_grid_arcs(grid)
        A, b, u = self._constructMatrix()
        super().__init__(A=A, b=b, u=u, use_sparse_matrix=True, minimize=True)

    def _constructMatrix(self):
        """
        Constructs the incidence matrix A, supply/demand vector b, and upper bound u
        for the shortest path problem.

        Returns:
            A (jnp.ndarray): Incidence matrix for flow conservation
            b (jnp.ndarray): Supply/demand vector
            u (jnp.ndarray): Upper bound for flow variables
        """
        # number of nodes and arcs
        num_nodes = self.grid[0] * self.grid[1]
        num_arcs = len(self.arcs)
        # construct incidence matrix A for flow conservation (build in numpy, then convert)
        A_np = np.zeros((num_nodes, num_arcs), dtype=np.float32)
        for arc_idx, (start, end) in enumerate(self.arcs):
            A_np[start, arc_idx] = 1
            A_np[end, arc_idx] = -1
        A = jnp.array(A_np)
        # construct supply/demand vector b
        b_np = np.zeros(num_nodes, dtype=np.float32)
        b_np[0] = 1            # source node (top-left) sends one unit
        b_np[num_nodes-1] = -1 # sink node (bottom-right) receives one unit
        b = jnp.array(b_np)
        # upper bound
        u = jnp.ones(len(self.arcs), dtype=jnp.float32)
        return A, b, u


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
