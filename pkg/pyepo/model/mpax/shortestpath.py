#!/usr/bin/env python
"""
Shortest path problem
"""

from __future__ import annotations

import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    pass

from pyepo.model.bases import shortestPathBase
from pyepo.model.mpax.mpaxmodel import optMpaxModel


class shortestPathModel(shortestPathBase, optMpaxModel):
    """
    MPAX-backed (JAX LP) shortest path on a grid network.

    Attributes:
        grid (tuple of int): Size of grid network
        arcs (list): List of arcs
    """

    use_sparse_matrix = True

    def _getModel(self) -> tuple:
        """
        Build MPAX matrices: equality flow-conservation A x = b, x in [0, 1].
        """
        num_nodes = self.grid[0] * self.grid[1]
        num_arcs = len(self.arcs)
        # node-arc incidence: +1 outgoing, -1 incoming
        A_np = np.zeros((num_nodes, num_arcs), dtype=np.float32)
        for arc_idx, (start, end) in enumerate(self.arcs):
            A_np[start, arc_idx] = 1
            A_np[end, arc_idx] = -1
        self.A = jnp.array(A_np)
        # supply / demand: source sends 1, sink receives 1
        b_np = np.zeros(num_nodes, dtype=np.float32)
        b_np[0] = 1
        b_np[num_nodes - 1] = -1
        self.b = jnp.array(b_np)
        # no inequality constraints
        self.G = jnp.zeros((0, num_arcs), dtype=jnp.float32)
        self.h = jnp.zeros((0,), dtype=jnp.float32)
        # variable bounds: x in [0, 1]
        self.l = jnp.zeros(num_arcs, dtype=jnp.float32)
        self.u = jnp.ones(num_arcs, dtype=jnp.float32)
        return None, []


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
