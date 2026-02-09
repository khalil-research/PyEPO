#!/usr/bin/env python
# coding: utf-8
"""
Tests for pyepo.utils: getArgs utility
"""

import pytest
import numpy as np

from pyepo.utils import getArgs

try:
    from pyepo.model.grb.knapsack import knapsackModel
    from pyepo.model.grb.shortestpath import shortestPathModel
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

requires_gurobi = pytest.mark.skipif(not _HAS_GUROBI, reason="Gurobi not installed")


# ============================================================
# getArgs tests
# ============================================================

@requires_gurobi
class TestGetArgs:

    def test_knapsack_args(self):
        weights = np.array([[3.0, 4.0, 5.0]])
        capacity = np.array([10.0])
        model = knapsackModel(weights=weights, capacity=capacity)
        args = getArgs(model)
        assert "weights" in args
        assert "capacity" in args
        np.testing.assert_array_equal(args["weights"], weights)
        np.testing.assert_array_equal(args["capacity"], capacity)

    def test_shortestpath_args(self):
        model = shortestPathModel(grid=(4, 4))
        args = getArgs(model)
        assert "grid" in args
        assert args["grid"] == (4, 4)

    def test_rebuild_from_args(self):
        """Args should be sufficient to rebuild the model."""
        model = shortestPathModel(grid=(3, 3))
        args = getArgs(model)
        model2 = shortestPathModel(**args)
        assert model2.num_cost == model.num_cost
        assert model2.grid == model.grid

    def test_rebuild_knapsack_from_args(self):
        weights = np.array([[3.0, 4.0, 5.0]])
        capacity = np.array([10.0])
        model = knapsackModel(weights=weights, capacity=capacity)
        args = getArgs(model)
        model2 = knapsackModel(**args)
        assert model2.num_cost == model.num_cost

    def test_no_extra_args(self):
        """getArgs should only return __init__ parameters, not internal state."""
        model = shortestPathModel(grid=(3, 3))
        args = getArgs(model)
        # 'arcs' is an attribute but NOT an __init__ parameter
        assert "arcs" not in args
        # '_model' is internal
        assert "_model" not in args

    def test_rebuilt_model_solves_same(self):
        """Rebuilt model should produce same solutions."""
        model = shortestPathModel(grid=(3, 3))
        args = getArgs(model)
        model2 = shortestPathModel(**args)
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        sol1, obj1 = model.solve()
        model2.setObj(cost)
        sol2, obj2 = model2.solve()
        np.testing.assert_allclose(obj1, obj2, atol=1e-6)
        np.testing.assert_allclose(sol1, sol2, atol=1e-6)
