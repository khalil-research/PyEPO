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
    from pyepo.model.grb.tsp import (
        tspGGModel, tspGGModelRel, tspDFJModel,
        tspMTZModel, tspMTZModelRel,
    )
    _HAS_GUROBI = True
except (ImportError, NameError):
    _HAS_GUROBI = False

try:
    from pyepo.model.omo.shortestpath import shortestPathModel as omoShortestPathModel
    from pyepo.model.omo.knapsack import knapsackModel as omoKnapsackModel
    _HAS_PYOMO = True
except (ImportError, NameError):
    _HAS_PYOMO = False

try:
    from pyepo.model.copt.shortestpath import shortestPathModel as coptShortestPathModel
    from pyepo.model.copt.knapsack import knapsackModel as coptKnapsackModel
    _HAS_COPT = True
except (ImportError, NameError):
    _HAS_COPT = False

requires_gurobi = pytest.mark.skipif(not _HAS_GUROBI, reason="Gurobi not installed")
requires_pyomo = pytest.mark.skipif(
    not (_HAS_PYOMO and _HAS_GUROBI),
    reason="Pyomo or Gurobi not installed"
)
requires_copt = pytest.mark.skipif(not _HAS_COPT, reason="COPT not installed")


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

    def test_copy(self, model):
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        model.setObj(cost)
        _, obj1 = model.solve()
        model_copy = model.copy()
        model_copy.setObj(cost)
        _, obj2 = model_copy.solve()
        np.testing.assert_allclose(obj1, obj2, atol=1e-6)

    def test_addConstr(self, model):
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        model.setObj(cost)
        _, obj1 = model.solve()
        # add a binding constraint: sum(x) <= 1
        model2 = model.addConstr(np.ones(model.num_cost), 1)
        model2.setObj(cost)
        _, obj2 = model2.solve()
        # added constraint should not improve objective (MAXIMIZE)
        assert obj2 <= obj1 + 1e-6


# ============================================================
# Pyomo: Shortest path model
# ============================================================

@requires_pyomo
class TestOmoShortestPathModel:

    @pytest.fixture
    def model(self):
        return omoShortestPathModel(grid=(3, 3), solver="gurobi")

    def test_init(self, model):
        assert model.grid == (3, 3)
        assert model.modelSense == EPO.MINIMIZE

    def test_num_cost(self, model):
        assert model.num_cost == 12

    def test_setObj_and_solve(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        sol, obj = model.solve()
        assert len(sol) == model.num_cost
        assert isinstance(obj, float)
        assert obj > 0

    def test_copy(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        _, obj1 = model.solve()
        model_copy = model.copy()
        model_copy.setObj(cost)
        _, obj2 = model_copy.solve()
        np.testing.assert_allclose(obj1, obj2, atol=1e-6)

    def test_addConstr(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        _, obj1 = model.solve()
        model2 = model.addConstr(np.ones(model.num_cost), 5)
        model2.setObj(cost)
        _, obj2 = model2.solve()
        assert obj2 >= obj1 - 1e-6


# ============================================================
# Pyomo: Knapsack model
# ============================================================

@requires_pyomo
class TestOmoKnapsackModel:

    @pytest.fixture
    def model(self):
        weights = np.array([[3.0, 4.0, 5.0, 6.0]])
        capacity = np.array([10.0])
        return omoKnapsackModel(weights=weights, capacity=capacity, solver="gurobi")

    def test_init(self, model):
        assert model.modelSense == EPO.MAXIMIZE

    def test_num_cost(self, model):
        assert model.num_cost == 4

    def test_solve(self, model):
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        model.setObj(cost)
        sol, obj = model.solve()
        sol = np.array(sol)
        assert np.allclose(sol, np.round(sol), atol=1e-6)

    def test_objective_value(self, model):
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        model.setObj(cost)
        sol, obj = model.solve()
        sol = np.array(sol)
        np.testing.assert_allclose(obj, np.dot(cost, sol), atol=1e-6)

    def test_copy(self, model):
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        model.setObj(cost)
        _, obj1 = model.solve()
        model_copy = model.copy()
        model_copy.setObj(cost)
        _, obj2 = model_copy.solve()
        np.testing.assert_allclose(obj1, obj2, atol=1e-6)

    def test_addConstr(self, model):
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        model.setObj(cost)
        _, obj1 = model.solve()
        model2 = model.addConstr(np.ones(model.num_cost), 1)
        model2.setObj(cost)
        _, obj2 = model2.solve()
        assert obj2 <= obj1 + 1e-6


# ============================================================
# COPT: Shortest path model
# ============================================================

@requires_copt
class TestCoptShortestPathModel:

    @pytest.fixture
    def model(self):
        return coptShortestPathModel(grid=(3, 3))

    def test_init(self, model):
        assert model.grid == (3, 3)
        assert model.modelSense == EPO.MINIMIZE

    def test_num_cost(self, model):
        assert model.num_cost == 12

    def test_setObj_and_solve(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        sol, obj = model.solve()
        assert len(sol) == model.num_cost
        assert isinstance(obj, float)
        assert obj > 0

    def test_copy(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        _, obj1 = model.solve()
        model_copy = model.copy()
        model_copy.setObj(cost)
        _, obj2 = model_copy.solve()
        np.testing.assert_allclose(obj1, obj2, atol=1e-6)

    def test_addConstr(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        _, obj1 = model.solve()
        model2 = model.addConstr(np.ones(model.num_cost), 5)
        model2.setObj(cost)
        _, obj2 = model2.solve()
        assert obj2 >= obj1 - 1e-6


# ============================================================
# COPT: Knapsack model
# ============================================================

@requires_copt
class TestCoptKnapsackModel:

    @pytest.fixture
    def model(self):
        weights = np.array([[3.0, 4.0, 5.0, 6.0]])
        capacity = np.array([10.0])
        return coptKnapsackModel(weights=weights, capacity=capacity)

    def test_init(self, model):
        assert model.modelSense == EPO.MAXIMIZE

    def test_num_cost(self, model):
        assert model.num_cost == 4

    def test_solve(self, model):
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        model.setObj(cost)
        sol, obj = model.solve()
        sol = np.array(sol)
        assert np.allclose(sol, np.round(sol), atol=1e-6)

    def test_copy(self, model):
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        model.setObj(cost)
        _, obj1 = model.solve()
        model_copy = model.copy()
        model_copy.setObj(cost)
        _, obj2 = model_copy.solve()
        np.testing.assert_allclose(obj1, obj2, atol=1e-6)

    def test_addConstr(self, model):
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        model.setObj(cost)
        _, obj1 = model.solve()
        model2 = model.addConstr(np.ones(model.num_cost), 1)
        model2.setObj(cost)
        _, obj2 = model2.solve()
        assert obj2 <= obj1 + 1e-6


# ============================================================
# TSP: Gavish-Graves (GG) formulation
# ============================================================

@requires_gurobi
class TestTspGGModel:

    @pytest.fixture
    def model(self):
        return tspGGModel(num_nodes=4)

    def test_init(self, model):
        assert model.num_nodes == 4
        assert model.modelSense == EPO.MINIMIZE

    def test_num_cost(self, model):
        # 4 nodes: C(4,2) = 6 edges
        assert model.num_cost == 6

    def test_edges_count(self, model):
        assert len(model.edges) == 6

    def test_setObj_and_solve(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        sol, obj = model.solve()
        sol = np.array(sol)
        assert len(sol) == model.num_cost
        assert isinstance(obj, float)
        # TSP solution should be binary
        assert np.allclose(sol, np.round(sol), atol=1e-6)

    def test_solution_is_tour(self, model):
        """Solution should form a valid Hamiltonian cycle."""
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        sol, _ = model.solve()
        sol = np.array(sol)
        # number of active edges should equal num_nodes
        assert int(np.sum(sol)) == model.num_nodes

    def test_getTour(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        sol, _ = model.solve()
        tour = model.getTour(sol)
        # tour should visit all nodes and return to start
        assert tour[0] == tour[-1]
        assert len(set(tour[:-1])) == model.num_nodes

    def test_setObj_wrong_size(self, model):
        with pytest.raises(ValueError):
            model.setObj(np.ones(3))

    def test_copy(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model_copy = model.copy()
        model_copy.setObj(cost)
        sol, obj = model_copy.solve()
        assert isinstance(obj, float)

    def test_addConstr(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        _, obj1 = model.solve()
        model2 = model.addConstr(np.ones(model.num_cost), 5)
        model2.setObj(cost)
        _, obj2 = model2.solve()
        # added constraint should not improve objective (MINIMIZE)
        assert obj2 >= obj1 - 1e-6

    def test_addConstr_infeasible(self, model):
        # 4-node TSP tour uses exactly 4 edges, so sum(edges) <= 3 is infeasible
        model2 = model.addConstr(np.ones(model.num_cost), 3)
        cost = np.random.RandomState(42).rand(model.num_cost)
        model2.setObj(cost)
        with pytest.raises(Exception):
            model2.solve()

    def test_relax(self, model):
        rel_model = model.relax()
        assert isinstance(rel_model, tspGGModelRel)
        cost = np.random.RandomState(42).rand(model.num_cost)
        rel_model.setObj(cost)
        sol, obj_rel = rel_model.solve()
        model.setObj(cost)
        _, obj_int = model.solve()
        # relaxed objective <= integer objective (MINIMIZE)
        assert obj_rel <= obj_int + 1e-6

    def test_relax_cannot_relax_again(self, model):
        rel_model = model.relax()
        with pytest.raises(RuntimeError):
            rel_model.relax()

    def test_relax_no_getTour(self, model):
        rel_model = model.relax()
        with pytest.raises(RuntimeError):
            rel_model.getTour([0] * model.num_cost)


# ============================================================
# TSP: DFJ formulation (lazy constraint generation)
# ============================================================

@requires_gurobi
class TestTspDFJModel:

    @pytest.fixture
    def model(self):
        return tspDFJModel(num_nodes=4)

    def test_init(self, model):
        assert model.num_nodes == 4
        assert model.modelSense == EPO.MINIMIZE

    def test_num_cost(self, model):
        assert model.num_cost == 6

    def test_setObj_and_solve(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        sol, obj = model.solve()
        sol = np.array(sol)
        assert len(sol) == model.num_cost
        assert isinstance(obj, float)
        assert np.allclose(sol, np.round(sol), atol=1e-6)

    def test_solution_is_tour(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        sol, _ = model.solve()
        sol = np.array(sol)
        assert int(np.sum(sol)) == model.num_nodes

    def test_getTour(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        sol, _ = model.solve()
        tour = model.getTour(sol)
        assert tour[0] == tour[-1]
        assert len(set(tour[:-1])) == model.num_nodes

    def test_copy(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model_copy = model.copy()
        model_copy.setObj(cost)
        sol, obj = model_copy.solve()
        assert isinstance(obj, float)

    def test_addConstr(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        _, obj1 = model.solve()
        model2 = model.addConstr(np.ones(model.num_cost), 5)
        model2.setObj(cost)
        _, obj2 = model2.solve()
        assert obj2 >= obj1 - 1e-6

    def test_addConstr_infeasible(self, model):
        # 4-node TSP tour uses exactly 4 edges, so sum(edges) <= 3 is infeasible
        model2 = model.addConstr(np.ones(model.num_cost), 3)
        cost = np.random.RandomState(42).rand(model.num_cost)
        model2.setObj(cost)
        with pytest.raises(Exception):
            model2.solve()


# ============================================================
# TSP: MTZ formulation
# ============================================================

@requires_gurobi
class TestTspMTZModel:

    @pytest.fixture
    def model(self):
        return tspMTZModel(num_nodes=4)

    def test_init(self, model):
        assert model.num_nodes == 4
        assert model.modelSense == EPO.MINIMIZE

    def test_num_cost(self, model):
        assert model.num_cost == 6

    def test_setObj_and_solve(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        sol, obj = model.solve()
        sol = np.array(sol)
        assert len(sol) == model.num_cost
        assert isinstance(obj, float)
        assert np.allclose(sol, np.round(sol), atol=1e-6)

    def test_solution_is_tour(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        sol, _ = model.solve()
        sol = np.array(sol)
        assert int(np.sum(sol)) == model.num_nodes

    def test_getTour(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        sol, _ = model.solve()
        tour = model.getTour(sol)
        assert tour[0] == tour[-1]
        assert len(set(tour[:-1])) == model.num_nodes

    def test_copy(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model_copy = model.copy()
        model_copy.setObj(cost)
        sol, obj = model_copy.solve()
        assert isinstance(obj, float)

    def test_addConstr(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        _, obj1 = model.solve()
        model2 = model.addConstr(np.ones(model.num_cost), 5)
        model2.setObj(cost)
        _, obj2 = model2.solve()
        assert obj2 >= obj1 - 1e-6

    def test_addConstr_infeasible(self, model):
        # 4-node TSP tour uses exactly 4 edges, so sum(edges) <= 3 is infeasible
        model2 = model.addConstr(np.ones(model.num_cost), 3)
        cost = np.random.RandomState(42).rand(model.num_cost)
        model2.setObj(cost)
        with pytest.raises(Exception):
            model2.solve()

    def test_relax(self, model):
        rel_model = model.relax()
        assert isinstance(rel_model, tspMTZModelRel)
        cost = np.random.RandomState(42).rand(model.num_cost)
        rel_model.setObj(cost)
        sol, obj_rel = rel_model.solve()
        model.setObj(cost)
        _, obj_int = model.solve()
        assert obj_rel <= obj_int + 1e-6

    def test_relax_cannot_relax_again(self, model):
        rel_model = model.relax()
        with pytest.raises(RuntimeError):
            rel_model.relax()


# ============================================================
# TSP: Cross-formulation consistency
# ============================================================

@requires_gurobi
class TestTspCrossFormulation:

    def test_gg_vs_dfj(self):
        cost = np.random.RandomState(42).rand(6)
        gg = tspGGModel(num_nodes=4)
        gg._model.Params.OutputFlag = 0
        gg.setObj(cost)
        _, gg_obj = gg.solve()
        dfj = tspDFJModel(num_nodes=4)
        dfj.setObj(cost)
        _, dfj_obj = dfj.solve()
        np.testing.assert_allclose(gg_obj, dfj_obj, atol=1e-4)

    def test_gg_vs_mtz(self):
        cost = np.random.RandomState(42).rand(6)
        gg = tspGGModel(num_nodes=4)
        gg._model.Params.OutputFlag = 0
        gg.setObj(cost)
        _, gg_obj = gg.solve()
        mtz = tspMTZModel(num_nodes=4)
        mtz.setObj(cost)
        _, mtz_obj = mtz.solve()
        np.testing.assert_allclose(gg_obj, mtz_obj, atol=1e-4)


# ============================================================
# Cross-backend consistency
# ============================================================

@requires_pyomo
class TestCrossBackendConsistency:

    def test_shortestpath_grb_vs_omo(self):
        cost = np.random.RandomState(42).rand(12)
        grb_model = shortestPathModel(grid=(3, 3))
        grb_model.setObj(cost)
        _, grb_obj = grb_model.solve()
        omo_model = omoShortestPathModel(grid=(3, 3), solver="gurobi")
        omo_model.setObj(cost)
        _, omo_obj = omo_model.solve()
        np.testing.assert_allclose(grb_obj, omo_obj, atol=1e-4)

    def test_knapsack_grb_vs_omo(self):
        weights = np.array([[3.0, 4.0, 5.0, 6.0]])
        capacity = np.array([10.0])
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        grb_model = knapsackModel(weights=weights, capacity=capacity)
        grb_model.setObj(cost)
        _, grb_obj = grb_model.solve()
        omo_model = omoKnapsackModel(weights=weights, capacity=capacity, solver="gurobi")
        omo_model.setObj(cost)
        _, omo_obj = omo_model.solve()
        np.testing.assert_allclose(grb_obj, omo_obj, atol=1e-4)
