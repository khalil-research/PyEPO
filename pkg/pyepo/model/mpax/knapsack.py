#!/usr/bin/env python
# coding: utf-8
"""
Knapsack problem
"""

try:
    import jax.numpy as jnp
    from mpax import create_lp, r2HPDHG
    _HAS_MPAX = True
except ImportError:
    _HAS_MPAX = False

from pyepo.model.mpax.mpaxmodel import optMpaxModel


class knapsackModel(optMpaxModel):
    """
    This class is optimization model for relexed knapsack problem

    Attributes:
        _model (GurobiPy model): Gurobi model
        weights (np.ndarray / list): Weights of items
        capacity (np.ndarray / listy): Total capacity
        items (list): List of item index
    """

    def __init__(self, weights, capacity):
        """
        Args:
            weights (np.ndarray / list): weights of items
            capacity (np.ndarray / list): total capacity
        """
        self.weights = weights
        self.capacity = capacity
        self.items = self.weights.shape[1]
        G, h, u = self._constructMatrix()
        super().__init__(G=G, h=h, u=u, use_sparse_matrix=False, minimize=False)

    def _constructMatrix(self):
        """
        Constructs the inequality constraint matrix G, right-hand side h, and
        upper bound u for the knapsack problem.

        Returns:
            G (jnp.ndarray): Weights of items
            h (jnp.ndarray): Total capacity
            u (jnp.ndarray): Upper bound for item selection
        """
        G = - jnp.array(self.weights, dtype=jnp.float32)
        h = - jnp.array(self.capacity, dtype=jnp.float32)
        u = jnp.ones(self.items, dtype=jnp.float32)
        return G, h, u


if __name__ == "__main__":
    import numpy as np

    # random seed
    np.random.seed(42)
    # set random cost for test
    cost = np.random.random(16)
    weights = np.random.choice(range(300, 800), size=(2,16)) / 100
    capacity = [20, 20]

    # solve model
    optmodel = knapsackModel(weights=weights, capacity=capacity) # init model
    optmodel = optmodel.copy()
    optmodel.setObj(cost) # set objective function
    sol, obj = optmodel.solve() # solve
    # print res
    print('Obj: {}'.format(obj))
    for i in range(16):
        if sol[i] > 1e-3:
            print(i)
