#!/usr/bin/env python
# coding: utf-8
"""
Shortest path problem
"""

try:
    import jax.numpy as jnp
    from mpax import create_lp, r2HPDHG
    _HAS_MPAX = True
except ImportError:
    _HAS_MPAX = False

from pyepo import EPO
from pyepo.model.opt import optModel

from pyepo.model.mpax.mpaxmodel import optMpaxModel


class shortestPathModel(optMpaxModel):
    """
    This class is optimization model for shortest path problem

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
        self.arcs = self._getArcs()
        A, b, u = self._constructMatrix()
        super().__init__(A=A, b=b, u=u, use_sparse_matrix=True, minimize=True)

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
        # construct incidence matrix A for flow conservation
        A = jnp.zeros((num_nodes, num_arcs), dtype=jnp.float32)
        for arc_idx, (start, end) in enumerate(self.arcs):
            A = A.at[start, arc_idx].set(1)
            A = A.at[end, arc_idx].set(-1)
        # construct supply/demand vector b
        b = jnp.zeros(num_nodes, dtype=jnp.float32)
        b = b.at[0].set(1)                  # source node (top-left) sends one unit
        b = b.at[num_nodes-1].set(-1)       # sink node (bottom-right) receives one unit
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
