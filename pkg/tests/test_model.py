#!/usr/bin/env python
# coding: utf-8
"""
Tests for pyepo.model: optimization models

These tests require Gurobi. Skip gracefully if not installed.
"""

import pytest
import numpy as np

from pyepo import EPO
from pyepo.model.opt import optModel

try:
    import gurobipy as gp
    from gurobipy import GRB
    from pyepo.model.grb.grbmodel import optGrbModel
    from pyepo.model.grb.knapsack import knapsackModel, knapsackModelRel
    from pyepo.model.grb.shortestpath import shortestPathModel
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

requires_gurobi = pytest.mark.skipif(not _HAS_GUROBI, reason="Gurobi not installed")


# ============================================================
# optModel base class
# ============================================================

class TestOptModelBase:

    def test_abstract_methods(self):
        """optModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            optModel()


# ============================================================
# Shortest path model
# ============================================================

@requires_gurobi
class TestShortestPathModel:

    @pytest.fixture
    def model(self):
        return shortestPathModel(grid=(3, 3))

    def test_init(self, model):
        assert model.grid == (3, 3)
        assert model.modelSense == EPO.MINIMIZE

    def test_num_cost(self, model):
        # 3x3 grid: horizontal edges = 3*2=6, vertical edges = 2*3=6, total=12
        assert model.num_cost == 12

    def test_arcs_count(self, model):
        assert len(model.arcs) == 12

    def test_setObj_and_solve(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        sol, obj = model.solve()
        assert len(sol) == model.num_cost
        assert isinstance(obj, float)
        assert obj > 0

    def test_setObj_wrong_size(self, model):
        with pytest.raises(ValueError):
            model.setObj(np.ones(5))

    def test_solution_is_path(self, model):
        """Solution should be a valid s-t flow (values in {0,1})."""
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        sol, _ = model.solve()
        sol = np.array(sol)
        # all values should be 0 or 1 (LP relaxation gives integer for shortest path)
        assert np.allclose(sol, np.round(sol), atol=1e-6)

    def test_copy(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model_copy = model.copy()
        model_copy.setObj(cost)
        sol, obj = model_copy.solve()
        assert isinstance(obj, float)

    def test_addConstr(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        # add a binding constraint
        model2 = model.addConstr(np.ones(model.num_cost), 5)
        model2.setObj(cost)
        sol2, obj2 = model2.solve()
        # original model
        model.setObj(cost)
        _, obj1 = model.solve()
        # added constraint should not improve objective
        assert obj2 >= obj1 - 1e-6

    def test_setObj_accepts_tensor(self, model):
        import torch
        cost = torch.rand(model.num_cost)
        model.setObj(cost)
        sol, obj = model.solve()
        assert isinstance(obj, float)

    def test_repr(self, model):
        r = repr(model)
        assert "shortestPathModel" in r

    def test_different_grid_sizes(self):
        for grid in [(2, 2), (3, 4), (5, 5)]:
            m = shortestPathModel(grid=grid)
            expected = (grid[0] - 1) * grid[1] + (grid[1] - 1) * grid[0]
            assert m.num_cost == expected


# ============================================================
# Knapsack model
# ============================================================

@requires_gurobi
class TestKnapsackModel:

    @pytest.fixture
    def model(self):
        weights = np.array([[3.0, 4.0, 5.0, 6.0]])
        capacity = np.array([10.0])
        return knapsackModel(weights=weights, capacity=capacity)

    def test_init(self, model):
        assert model.modelSense == EPO.MAXIMIZE
        assert model.items == 4

    def test_num_cost(self, model):
        assert model.num_cost == 4

    def test_solve(self, model):
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        model.setObj(cost)
        sol, obj = model.solve()
        sol = np.array(sol)
        # check solution is binary
        assert np.allclose(sol, np.round(sol), atol=1e-6)
        # check feasibility: weights @ sol <= capacity
        assert np.all(model.weights @ sol <= model.capacity + 1e-6)

    def test_objective_value(self, model):
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        model.setObj(cost)
        sol, obj = model.solve()
        sol = np.array(sol)
        np.testing.assert_allclose(obj, np.dot(cost, sol), atol=1e-6)

    def test_setObj_wrong_size(self, model):
        with pytest.raises(ValueError):
            model.setObj(np.ones(10))

    def test_relax(self, model):
        rel_model = model.relax()
        assert isinstance(rel_model, knapsackModelRel)
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        rel_model.setObj(cost)
        sol, obj_rel = rel_model.solve()
        # relaxed objective >= integer objective
        model.setObj(cost)
        _, obj_int = model.solve()
        assert obj_rel >= obj_int - 1e-6

    def test_relax_cannot_relax_again(self, model):
        rel_model = model.relax()
        with pytest.raises(RuntimeError):
            rel_model.relax()

    def test_multidim_knapsack(self):
        weights = np.array([[3.0, 4.0, 5.0], [2.0, 3.0, 4.0]])
        capacity = np.array([8.0, 7.0])
        model = knapsackModel(weights=weights, capacity=capacity)
        cost = np.array([5.0, 4.0, 3.0])
        model.setObj(cost)
        sol, obj = model.solve()
        sol = np.array(sol)
        assert np.all(weights @ sol <= capacity + 1e-6)
