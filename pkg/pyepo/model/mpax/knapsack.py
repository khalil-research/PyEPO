#!/usr/bin/env python
"""
Knapsack problem
"""

from __future__ import annotations

import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    pass

from pyepo.model.bases import knapsackBase
from pyepo.model.mpax.mpaxmodel import optMpaxModel


class knapsackModel(knapsackBase, optMpaxModel):
    """
    MPAX-backed (JAX LP) knapsack — LP relaxation.

    Attributes:
        _model: MPAX model
        weights (np.ndarray): Weights of items
        capacity (np.ndarray): Total capacity
        items (list): List of item index
    """

    use_sparse_matrix = False

    def _getModel(self) -> tuple:
        """
        Build MPAX matrices: inequality G x >= h with G = -W, h = -c,
        which encodes W x <= c. Variables relaxed to x in [0, 1].
        """
        num_items = self.weights.shape[1]
        # no equality constraints
        self.A = jnp.zeros((0, num_items), dtype=jnp.float32)
        self.b = jnp.zeros((0,), dtype=jnp.float32)
        # inequality: -W x >= -c  (equivalent to W x <= c)
        self.G = -jnp.array(self.weights, dtype=jnp.float32)
        self.h = -jnp.array(np.asarray(self.capacity), dtype=jnp.float32)
        # variable bounds: x in [0, 1]
        self.l = jnp.zeros(num_items, dtype=jnp.float32)
        self.u = jnp.ones(num_items, dtype=jnp.float32)
        return None, []
