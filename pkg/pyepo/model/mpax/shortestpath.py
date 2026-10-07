#!/usr/bin/env python
"""
Shortest path problem
"""

from __future__ import annotations

import numpy as np

try:
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO
except ImportError:
    jnp = None

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
        endpoints = np.asarray(self.arcs, dtype=np.int32).reshape(-1)
        indices = np.column_stack((endpoints, np.repeat(np.arange(num_arcs), 2)))
        values = np.tile(np.array([1, -1], dtype=np.float32), num_arcs)
        self.A = BCOO(
            (jnp.asarray(values), jnp.asarray(indices, dtype=jnp.int32)),
            shape=(num_nodes, num_arcs),
            unique_indices=True,
        )
        if not self.use_sparse_matrix:
            self.A = self.A.todense()
        # supply / demand: source sends 1, sink receives 1
        b_np = np.zeros(num_nodes, dtype=np.float32)
        b_np[0] = 1
        b_np[num_nodes - 1] = -1
        self.b = jnp.array(b_np)
        # no inequality constraints
        self.G = (
            BCOO.fromdense(jnp.zeros((0, num_arcs), dtype=jnp.float32), nse=0)
            if self.use_sparse_matrix
            else jnp.zeros((0, num_arcs), dtype=jnp.float32)
        )
        self.h = jnp.zeros((0,), dtype=jnp.float32)
        # variable bounds: x in [0, 1]
        self.l = jnp.zeros(num_arcs, dtype=jnp.float32)
        self.u = jnp.ones(num_arcs, dtype=jnp.float32)
        return None, []
