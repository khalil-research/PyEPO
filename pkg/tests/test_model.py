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

from pyepo.data.portfolio import genData

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
    from pyepo.model.grb.vrp import (
        vrpRCIModel, vrpMTZModel, vrpMTZModelRel,
    )
    from pyepo.model.grb.portfolio import portfolioModel
    _HAS_GUROBI = True
except (ImportError, NameError):
    _HAS_GUROBI = False

try:
    # probe pyomo directly: `from pyepo.model.omo.*` succeeds even when pyomo is missing
    import pyomo.environ  # noqa: F401
    from pyepo.model.omo.shortestpath import shortestPathModel as omoShortestPathModel
    from pyepo.model.omo.knapsack import (
        knapsackModel as omoKnapsackModel,
        knapsackModelRel as omoKnapsackModelRel,
    )
    from pyepo.model.omo.tsp import (
        tspGGModel as omoTspGGModel, tspGGModelRel as omoTspGGModelRel,
        tspMTZModel as omoTspMTZModel, tspMTZModelRel as omoTspMTZModelRel,
    )
    from pyepo.model.omo.vrp import (
        vrpMTZModel as omoVrpMTZModel,
        vrpMTZModelRel as omoVrpMTZModelRel,
    )
    from pyepo.model.omo.portfolio import portfolioModel as omoPortfolioModel
    _HAS_PYOMO = True
except (ImportError, NameError):
    _HAS_PYOMO = False

try:
    from pyepo.model.copt.shortestpath import shortestPathModel as coptShortestPathModel
    from pyepo.model.copt.knapsack import (
        knapsackModel as coptKnapsackModel,
        knapsackModelRel as coptKnapsackModelRel,
    )
    from pyepo.model.copt.tsp import (
        tspGGModel as coptTspGGModel, tspGGModelRel as coptTspGGModelRel,
        tspDFJModel as coptTspDFJModel,
        tspMTZModel as coptTspMTZModel, tspMTZModelRel as coptTspMTZModelRel,
    )
    from pyepo.model.copt.vrp import (
        vrpRCIModel as coptVrpRCIModel,
        vrpMTZModel as coptVrpMTZModel,
        vrpMTZModelRel as coptVrpMTZModelRel,
    )
    from pyepo.model.copt.portfolio import portfolioModel as coptPortfolioModel
    _HAS_COPT = True
except (ImportError, NameError):
    _HAS_COPT = False

try:
    from pyepo.model.ort.ortmodel import _HAS_ORTOOLS
    if _HAS_ORTOOLS:
        from pyepo.model.ort.shortestpath import shortestPathModel as ortShortestPathModel
        from pyepo.model.ort.shortestpath import shortestPathCpModel as ortShortestPathCpModel
        from pyepo.model.ort.knapsack import knapsackModel as ortKnapsackModel
        from pyepo.model.ort.knapsack import knapsackModelRel as ortKnapsackModelRel
        from pyepo.model.ort.knapsack import knapsackCpModel as ortKnapsackCpModel
except (ImportError, NameError):
    _HAS_ORTOOLS = False

try:
    import jax  # noqa: F401
    import mpax  # noqa: F401
    from pyepo.model.mpax.shortestpath import shortestPathModel as mpaxShortestPathModel
    from pyepo.model.mpax.knapsack import knapsackModel as mpaxKnapsackModel
    _HAS_MPAX = True
except (ImportError, NameError):
    _HAS_MPAX = False

requires_gurobi = pytest.mark.skipif(not _HAS_GUROBI, reason="Gurobi not installed")
requires_pyomo = pytest.mark.skipif(
    not (_HAS_PYOMO and _HAS_GUROBI),
    reason="Pyomo or Gurobi not installed"
)
requires_copt = pytest.mark.skipif(not _HAS_COPT, reason="COPT not installed")
requires_ortools = pytest.mark.skipif(not _HAS_ORTOOLS, reason="OR-Tools not installed")
requires_mpax = pytest.mark.skipif(not _HAS_MPAX, reason="MPAX (jax + mpax) not installed")
requires_mpax_gurobi = pytest.mark.skipif(
    not (_HAS_MPAX and _HAS_GUROBI),
    reason="MPAX or Gurobi not installed"
)


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
        assert model.items == [0, 1, 2, 3]

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

    def test_relax(self, model):
        rel_model = model.relax()
        assert isinstance(rel_model, omoKnapsackModelRel)
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        rel_model.setObj(cost)
        _, obj_rel = rel_model.solve()
        model.setObj(cost)
        _, obj_int = model.solve()
        assert obj_rel >= obj_int - 1e-6


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

    def test_relax(self, model):
        rel_model = model.relax()
        assert isinstance(rel_model, coptKnapsackModelRel)
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        rel_model.setObj(cost)
        _, obj_rel = rel_model.solve()
        model.setObj(cost)
        _, obj_int = model.solve()
        assert obj_rel >= obj_int - 1e-6


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

    def test_portfolio_grb_vs_omo(self):
        cov, _, revenue = genData(num_data=10, num_features=4, num_assets=10, deg=1)
        cost = revenue[0]
        grb_model = portfolioModel(num_assets=10, covariance=cov)
        grb_model.setObj(cost)
        _, grb_obj = grb_model.solve()
        omo_model = omoPortfolioModel(num_assets=10, covariance=cov, solver="gurobi")
        omo_model.setObj(cost)
        _, omo_obj = omo_model.solve()
        np.testing.assert_allclose(grb_obj, omo_obj, atol=1e-4)

    def test_tsp_gg_grb_vs_omo(self):
        cost = np.random.RandomState(42).rand(6)
        grb_model = tspGGModel(num_nodes=4)
        grb_model._model.Params.OutputFlag = 0
        grb_model.setObj(cost)
        _, grb_obj = grb_model.solve()
        omo_model = omoTspGGModel(num_nodes=4, solver="gurobi")
        omo_model.setObj(cost)
        _, omo_obj = omo_model.solve()
        np.testing.assert_allclose(grb_obj, omo_obj, atol=1e-4)

    def test_tsp_mtz_grb_vs_omo(self):
        cost = np.random.RandomState(42).rand(6)
        grb_model = tspMTZModel(num_nodes=4)
        grb_model.setObj(cost)
        _, grb_obj = grb_model.solve()
        omo_model = omoTspMTZModel(num_nodes=4, solver="gurobi")
        omo_model.setObj(cost)
        _, omo_obj = omo_model.solve()
        np.testing.assert_allclose(grb_obj, omo_obj, atol=1e-4)

    def test_shortestpath_solution_matches(self):
        cost = np.random.RandomState(42).rand(12)
        grb_model = shortestPathModel(grid=(3, 3))
        grb_model.setObj(cost)
        grb_sol, _ = grb_model.solve()
        omo_model = omoShortestPathModel(grid=(3, 3), solver="gurobi")
        omo_model.setObj(cost)
        omo_sol, _ = omo_model.solve()
        np.testing.assert_allclose(np.asarray(grb_sol), np.asarray(omo_sol), atol=1e-4)

    def test_knapsack_solution_matches(self):
        weights = np.array([[3.0, 4.0, 5.0, 6.0]])
        capacity = np.array([10.0])
        # cost yields unique optimum [1, 1, 0, 0]
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        grb_model = knapsackModel(weights=weights, capacity=capacity)
        grb_model.setObj(cost)
        grb_sol, _ = grb_model.solve()
        omo_model = omoKnapsackModel(weights=weights, capacity=capacity, solver="gurobi")
        omo_model.setObj(cost)
        omo_sol, _ = omo_model.solve()
        np.testing.assert_allclose(np.asarray(grb_sol), np.asarray(omo_sol), atol=1e-4)


# ============================================================
# Portfolio model
# ============================================================

@requires_gurobi
class TestPortfolioModel:

    @pytest.fixture
    def model(self):
        cov, _, _ = genData(num_data=10, num_features=4, num_assets=10, deg=1)
        return portfolioModel(num_assets=10, covariance=cov)

    def test_init(self, model):
        assert model.modelSense == EPO.MAXIMIZE
        assert model.num_assets == 10

    def test_num_cost(self, model):
        assert model.num_cost == 10

    def test_solve(self, model):
        cov, _, revenue = genData(num_data=10, num_features=4, num_assets=10, deg=1)
        model.setObj(revenue[0])
        sol, obj = model.solve()
        sol = np.array(sol)
        assert len(sol) == model.num_cost
        assert isinstance(obj, float)
        # budget constraint: sum(x) == 1
        np.testing.assert_allclose(np.sum(sol), 1.0, atol=1e-4)

    def test_copy(self, model):
        cov, _, revenue = genData(num_data=10, num_features=4, num_assets=10, deg=1)
        model.setObj(revenue[0])
        _, obj1 = model.solve()
        model_copy = model.copy()
        model_copy.setObj(revenue[0])
        _, obj2 = model_copy.solve()
        np.testing.assert_allclose(obj1, obj2, atol=1e-6)

    def test_addConstr(self, model):
        cov, _, revenue = genData(num_data=10, num_features=4, num_assets=10, deg=1)
        model.setObj(revenue[0])
        _, obj1 = model.solve()
        coefs = np.zeros(model.num_cost)
        coefs[:3] = 1.0
        model2 = model.addConstr(coefs, 0.5)
        model2.setObj(revenue[0])
        _, obj2 = model2.solve()
        # added constraint should not improve objective (MAXIMIZE)
        assert obj2 <= obj1 + 1e-6


# ============================================================
# COPT: Portfolio model
# ============================================================

@requires_copt
class TestCoptPortfolioModel:

    @pytest.fixture
    def model(self):
        cov, _, _ = genData(num_data=10, num_features=4, num_assets=10, deg=1)
        return coptPortfolioModel(num_assets=10, covariance=cov)

    def test_init(self, model):
        assert model.modelSense == EPO.MAXIMIZE
        assert model.num_assets == 10

    def test_num_cost(self, model):
        assert model.num_cost == 10

    def test_solve(self, model):
        cov, _, revenue = genData(num_data=10, num_features=4, num_assets=10, deg=1)
        model.setObj(revenue[0])
        sol, obj = model.solve()
        sol = np.array(sol)
        assert len(sol) == model.num_cost
        assert isinstance(obj, float)
        np.testing.assert_allclose(np.sum(sol), 1.0, atol=1e-4)

    def test_copy(self, model):
        cov, _, revenue = genData(num_data=10, num_features=4, num_assets=10, deg=1)
        model.setObj(revenue[0])
        _, obj1 = model.solve()
        model_copy = model.copy()
        model_copy.setObj(revenue[0])
        _, obj2 = model_copy.solve()
        np.testing.assert_allclose(obj1, obj2, atol=1e-6)

    def test_addConstr(self, model):
        cov, _, revenue = genData(num_data=10, num_features=4, num_assets=10, deg=1)
        model.setObj(revenue[0])
        _, obj1 = model.solve()
        coefs = np.zeros(model.num_cost)
        coefs[:3] = 1.0
        model2 = model.addConstr(coefs, 0.5)
        model2.setObj(revenue[0])
        _, obj2 = model2.solve()
        assert obj2 <= obj1 + 1e-6


# ============================================================
# Pyomo: Portfolio model
# ============================================================

@requires_pyomo
class TestOmoPortfolioModel:

    @pytest.fixture
    def model(self):
        cov, _, _ = genData(num_data=10, num_features=4, num_assets=10, deg=1)
        return omoPortfolioModel(num_assets=10, covariance=cov, solver="gurobi")

    def test_init(self, model):
        assert model.modelSense == EPO.MAXIMIZE
        assert model.num_assets == 10

    def test_num_cost(self, model):
        assert model.num_cost == 10

    def test_solve(self, model):
        cov, _, revenue = genData(num_data=10, num_features=4, num_assets=10, deg=1)
        model.setObj(revenue[0])
        sol, obj = model.solve()
        sol = np.array(sol)
        assert len(sol) == model.num_cost
        assert isinstance(obj, float)
        np.testing.assert_allclose(np.sum(sol), 1.0, atol=1e-4)

    def test_copy(self, model):
        cov, _, revenue = genData(num_data=10, num_features=4, num_assets=10, deg=1)
        model.setObj(revenue[0])
        _, obj1 = model.solve()
        model_copy = model.copy()
        model_copy.setObj(revenue[0])
        _, obj2 = model_copy.solve()
        np.testing.assert_allclose(obj1, obj2, atol=1e-6)

    def test_addConstr(self, model):
        cov, _, revenue = genData(num_data=10, num_features=4, num_assets=10, deg=1)
        model.setObj(revenue[0])
        _, obj1 = model.solve()
        coefs = np.zeros(model.num_cost)
        coefs[:3] = 1.0
        model2 = model.addConstr(coefs, 0.5)
        model2.setObj(revenue[0])
        _, obj2 = model2.solve()
        assert obj2 <= obj1 + 1e-6


# ============================================================
# COPT: TSP Gavish-Graves (GG) formulation
# ============================================================

@requires_copt
class TestCoptTspGGModel:

    @pytest.fixture
    def model(self):
        return coptTspGGModel(num_nodes=4)

    def test_init(self, model):
        assert model.num_nodes == 4
        assert model.modelSense == EPO.MINIMIZE

    def test_num_cost(self, model):
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
        model2 = model.addConstr(np.ones(model.num_cost), 3)
        cost = np.random.RandomState(42).rand(model.num_cost)
        model2.setObj(cost)
        with pytest.raises(Exception):
            model2.solve()

    def test_relax(self, model):
        rel_model = model.relax()
        assert isinstance(rel_model, coptTspGGModelRel)
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

    def test_relax_no_getTour(self, model):
        rel_model = model.relax()
        with pytest.raises(RuntimeError):
            rel_model.getTour([0] * model.num_cost)


# ============================================================
# COPT: TSP DFJ formulation (lazy constraint generation)
# ============================================================

@requires_copt
class TestCoptTspDFJModel:

    @pytest.fixture
    def model(self):
        return coptTspDFJModel(num_nodes=4)

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
        model2 = model.addConstr(np.ones(model.num_cost), 3)
        cost = np.random.RandomState(42).rand(model.num_cost)
        model2.setObj(cost)
        with pytest.raises(Exception):
            model2.solve()


# ============================================================
# COPT: TSP MTZ formulation
# ============================================================

@requires_copt
class TestCoptTspMTZModel:

    @pytest.fixture
    def model(self):
        return coptTspMTZModel(num_nodes=4)

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
        model2 = model.addConstr(np.ones(model.num_cost), 3)
        cost = np.random.RandomState(42).rand(model.num_cost)
        model2.setObj(cost)
        with pytest.raises(Exception):
            model2.solve()

    def test_relax(self, model):
        rel_model = model.relax()
        assert isinstance(rel_model, coptTspMTZModelRel)
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
# COPT: TSP Cross-formulation consistency
# ============================================================

@requires_copt
class TestCoptTspCrossFormulation:

    def test_gg_vs_dfj(self):
        cost = np.random.RandomState(42).rand(6)
        gg = coptTspGGModel(num_nodes=4)
        gg.setObj(cost)
        _, gg_obj = gg.solve()
        dfj = coptTspDFJModel(num_nodes=4)
        dfj.setObj(cost)
        _, dfj_obj = dfj.solve()
        np.testing.assert_allclose(gg_obj, dfj_obj, atol=1e-4)

    def test_gg_vs_mtz(self):
        cost = np.random.RandomState(42).rand(6)
        gg = coptTspGGModel(num_nodes=4)
        gg.setObj(cost)
        _, gg_obj = gg.solve()
        mtz = coptTspMTZModel(num_nodes=4)
        mtz.setObj(cost)
        _, mtz_obj = mtz.solve()
        np.testing.assert_allclose(gg_obj, mtz_obj, atol=1e-4)


# ============================================================
# Pyomo: TSP Gavish-Graves (GG) formulation
# ============================================================

@requires_pyomo
class TestOmoTspGGModel:

    @pytest.fixture
    def model(self):
        return omoTspGGModel(num_nodes=4, solver="gurobi")

    def test_init(self, model):
        assert model.num_nodes == 4
        assert model.modelSense == EPO.MINIMIZE

    def test_num_cost(self, model):
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
        model2 = model.addConstr(np.ones(model.num_cost), 3)
        cost = np.random.RandomState(42).rand(model.num_cost)
        model2.setObj(cost)
        with pytest.raises(Exception):
            model2.solve()

    def test_relax(self, model):
        rel_model = model.relax()
        assert isinstance(rel_model, omoTspGGModelRel)
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

    def test_relax_no_getTour(self, model):
        rel_model = model.relax()
        with pytest.raises(RuntimeError):
            rel_model.getTour([0] * model.num_cost)


# ============================================================
# Pyomo: TSP MTZ formulation
# ============================================================

@requires_pyomo
class TestOmoTspMTZModel:

    @pytest.fixture
    def model(self):
        return omoTspMTZModel(num_nodes=4, solver="gurobi")

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
        model2 = model.addConstr(np.ones(model.num_cost), 3)
        cost = np.random.RandomState(42).rand(model.num_cost)
        model2.setObj(cost)
        with pytest.raises(Exception):
            model2.solve()

    def test_relax(self, model):
        rel_model = model.relax()
        assert isinstance(rel_model, omoTspMTZModelRel)
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
# Pyomo: TSP Cross-formulation consistency
# ============================================================

@requires_pyomo
class TestOmoTspCrossFormulation:

    def test_gg_vs_mtz(self):
        cost = np.random.RandomState(42).rand(6)
        gg = omoTspGGModel(num_nodes=4, solver="gurobi")
        gg.setObj(cost)
        _, gg_obj = gg.solve()
        mtz = omoTspMTZModel(num_nodes=4, solver="gurobi")
        mtz.setObj(cost)
        _, mtz_obj = mtz.solve()
        np.testing.assert_allclose(gg_obj, mtz_obj, atol=1e-4)


# ============================================================
# Cross-backend consistency: COPT
# ============================================================

@pytest.mark.skipif(not (_HAS_GUROBI and _HAS_COPT), reason="Gurobi or COPT not installed")
class TestCrossBackendCopt:

    def test_shortestpath_grb_vs_copt(self):
        cost = np.random.RandomState(42).rand(12)
        grb_model = shortestPathModel(grid=(3, 3))
        grb_model.setObj(cost)
        _, grb_obj = grb_model.solve()
        copt_model = coptShortestPathModel(grid=(3, 3))
        copt_model.setObj(cost)
        _, copt_obj = copt_model.solve()
        np.testing.assert_allclose(grb_obj, copt_obj, atol=1e-4)

    def test_knapsack_grb_vs_copt(self):
        weights = np.array([[3.0, 4.0, 5.0, 6.0]])
        capacity = np.array([10.0])
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        grb_model = knapsackModel(weights=weights, capacity=capacity)
        grb_model.setObj(cost)
        _, grb_obj = grb_model.solve()
        copt_model = coptKnapsackModel(weights=weights, capacity=capacity)
        copt_model.setObj(cost)
        _, copt_obj = copt_model.solve()
        np.testing.assert_allclose(grb_obj, copt_obj, atol=1e-4)

    def test_portfolio_grb_vs_copt(self):
        cov, _, revenue = genData(num_data=10, num_features=4, num_assets=10, deg=1)
        cost = revenue[0]
        grb_model = portfolioModel(num_assets=10, covariance=cov)
        grb_model.setObj(cost)
        _, grb_obj = grb_model.solve()
        copt_model = coptPortfolioModel(num_assets=10, covariance=cov)
        copt_model.setObj(cost)
        _, copt_obj = copt_model.solve()
        np.testing.assert_allclose(grb_obj, copt_obj, atol=1e-4)

    def test_tsp_gg_grb_vs_copt(self):
        cost = np.random.RandomState(42).rand(6)
        grb_model = tspGGModel(num_nodes=4)
        grb_model._model.Params.OutputFlag = 0
        grb_model.setObj(cost)
        _, grb_obj = grb_model.solve()
        copt_model = coptTspGGModel(num_nodes=4)
        copt_model.setObj(cost)
        _, copt_obj = copt_model.solve()
        np.testing.assert_allclose(grb_obj, copt_obj, atol=1e-4)

    def test_tsp_mtz_grb_vs_copt(self):
        cost = np.random.RandomState(42).rand(6)
        grb_model = tspMTZModel(num_nodes=4)
        grb_model.setObj(cost)
        _, grb_obj = grb_model.solve()
        copt_model = coptTspMTZModel(num_nodes=4)
        copt_model.setObj(cost)
        _, copt_obj = copt_model.solve()
        np.testing.assert_allclose(grb_obj, copt_obj, atol=1e-4)


# ============================================================
# OR-Tools pywraplp: Shortest path model
# ============================================================

@requires_ortools
class TestOrtShortestPathModel:

    @pytest.fixture
    def model(self):
        return ortShortestPathModel(grid=(3, 3))

    def test_init(self, model):
        assert model.grid == (3, 3)
        assert model.modelSense == EPO.MINIMIZE

    def test_num_cost(self, model):
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
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        sol, _ = model.solve()
        sol = np.array(sol)
        assert np.allclose(sol, np.round(sol), atol=1e-6)

    def test_copy(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model_copy = model.copy()
        model_copy.setObj(cost)
        sol, obj = model_copy.solve()
        assert isinstance(obj, float)

    def test_addConstr(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model2 = model.addConstr(np.ones(model.num_cost), 5)
        model2.setObj(cost)
        _, obj2 = model2.solve()
        model.setObj(cost)
        _, obj1 = model.solve()
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
            m = ortShortestPathModel(grid=grid)
            expected = (grid[0] - 1) * grid[1] + (grid[1] - 1) * grid[0]
            assert m.num_cost == expected


# ============================================================
# OR-Tools pywraplp: Knapsack model
# ============================================================

@requires_ortools
class TestOrtKnapsackModel:

    @pytest.fixture
    def model(self):
        weights = np.array([[3.0, 4.0, 5.0, 6.0]])
        capacity = np.array([10.0])
        return ortKnapsackModel(weights=weights, capacity=capacity)

    def test_init(self, model):
        assert model.modelSense == EPO.MAXIMIZE
        assert model.items == [0, 1, 2, 3]

    def test_num_cost(self, model):
        assert model.num_cost == 4

    def test_solve(self, model):
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        model.setObj(cost)
        sol, obj = model.solve()
        sol = np.array(sol)
        assert np.allclose(sol, np.round(sol), atol=1e-6)
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
        assert isinstance(rel_model, ortKnapsackModelRel)
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        rel_model.setObj(cost)
        _, obj_rel = rel_model.solve()
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
        model = ortKnapsackModel(weights=weights, capacity=capacity)
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
        model2 = model.addConstr(np.ones(model.num_cost), 1)
        model2.setObj(cost)
        _, obj2 = model2.solve()
        assert obj2 <= obj1 + 1e-6


# ============================================================
# OR-Tools CP-SAT: Shortest path model
# ============================================================

@requires_ortools
class TestOrtCpShortestPathModel:

    @pytest.fixture
    def model(self):
        return ortShortestPathCpModel(grid=(3, 3))

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

    def test_solution_is_path(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        sol, _ = model.solve()
        sol = np.array(sol)
        assert np.allclose(sol, np.round(sol), atol=1e-6)

    def test_copy(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model_copy = model.copy()
        model_copy.setObj(cost)
        sol, obj = model_copy.solve()
        assert isinstance(obj, float)

    def test_addConstr(self, model):
        cost = np.random.RandomState(42).rand(model.num_cost)
        model2 = model.addConstr(np.ones(model.num_cost), 5)
        model2.setObj(cost)
        _, obj2 = model2.solve()
        model.setObj(cost)
        _, obj1 = model.solve()
        assert obj2 >= obj1 - 1e-6

    def test_relax_not_supported(self, model):
        with pytest.raises(RuntimeError):
            model.relax()


# ============================================================
# OR-Tools CP-SAT: Knapsack model
# ============================================================

@requires_ortools
class TestOrtCpKnapsackModel:

    @pytest.fixture
    def model(self):
        weights = np.array([[3, 4, 5, 6]])
        capacity = np.array([10])
        return ortKnapsackCpModel(weights=weights, capacity=capacity)

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
        np.testing.assert_allclose(obj, np.dot(cost, sol), atol=1e-2)

    def test_copy(self, model):
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        model.setObj(cost)
        _, obj1 = model.solve()
        model_copy = model.copy()
        model_copy.setObj(cost)
        _, obj2 = model_copy.solve()
        np.testing.assert_allclose(obj1, obj2, atol=1e-2)

    def test_addConstr(self, model):
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        model.setObj(cost)
        _, obj1 = model.solve()
        model2 = model.addConstr(np.ones(model.num_cost), 1)
        model2.setObj(cost)
        _, obj2 = model2.solve()
        assert obj2 <= obj1 + 1e-6

    def test_float_weights_raise(self):
        weights = np.array([[3.5, 4.0, 5.0, 6.0]])
        capacity = np.array([10])
        with pytest.raises(ValueError, match="integer weights"):
            ortKnapsackCpModel(weights=weights, capacity=capacity)

    def test_float_capacity_raise(self):
        weights = np.array([[3, 4, 5, 6]])
        capacity = np.array([10.5])
        with pytest.raises(ValueError, match="integer capacity"):
            ortKnapsackCpModel(weights=weights, capacity=capacity)

    def test_relax_not_supported(self, model):
        with pytest.raises(RuntimeError):
            model.relax()


# ============================================================
# Cross-backend consistency: OR-Tools
# ============================================================

@pytest.mark.skipif(not (_HAS_GUROBI and _HAS_ORTOOLS), reason="Gurobi or OR-Tools not installed")
class TestCrossBackendOrt:

    def test_shortestpath_grb_vs_ort(self):
        cost = np.random.RandomState(42).rand(12)
        grb_model = shortestPathModel(grid=(3, 3))
        grb_model.setObj(cost)
        _, grb_obj = grb_model.solve()
        ort_model = ortShortestPathModel(grid=(3, 3))
        ort_model.setObj(cost)
        _, ort_obj = ort_model.solve()
        np.testing.assert_allclose(grb_obj, ort_obj, atol=1e-4)

    def test_knapsack_grb_vs_ort(self):
        weights = np.array([[3.0, 4.0, 5.0, 6.0]])
        capacity = np.array([10.0])
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        grb_model = knapsackModel(weights=weights, capacity=capacity)
        grb_model.setObj(cost)
        _, grb_obj = grb_model.solve()
        ort_model = ortKnapsackModel(weights=weights, capacity=capacity)
        ort_model.setObj(cost)
        _, ort_obj = ort_model.solve()
        np.testing.assert_allclose(grb_obj, ort_obj, atol=1e-4)

    def test_shortestpath_grb_vs_ort_cpsat(self):
        cost = np.random.RandomState(42).rand(12)
        grb_model = shortestPathModel(grid=(3, 3))
        grb_model.setObj(cost)
        _, grb_obj = grb_model.solve()
        ort_model = ortShortestPathCpModel(grid=(3, 3))
        ort_model.setObj(cost)
        _, ort_obj = ort_model.solve()
        np.testing.assert_allclose(grb_obj, ort_obj, atol=1e-2)

    def test_knapsack_grb_vs_ort_cpsat(self):
        weights = np.array([[3.0, 4.0, 5.0, 6.0]])
        capacity = np.array([10.0])
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        grb_model = knapsackModel(weights=weights, capacity=capacity)
        grb_model.setObj(cost)
        _, grb_obj = grb_model.solve()
        ort_model = ortKnapsackCpModel(weights=weights, capacity=capacity)
        ort_model.setObj(cost)
        _, ort_obj = ort_model.solve()
        np.testing.assert_allclose(grb_obj, ort_obj, atol=1e-2)


# ============================================================
# MPAX (JAX-based LP) backend
# ============================================================
# MPAX is PDHG-based (tol ~1e-4); knapsack here is LP relaxation, compare vs knapsackModelRel.

@requires_mpax
class TestMpaxShortestPathModel:

    @pytest.fixture
    def model(self):
        return mpaxShortestPathModel(grid=(3, 3))

    def test_init(self, model):
        assert model.modelSense == EPO.MINIMIZE
        assert model.grid == (3, 3)

    def test_num_cost(self, model):
        # 3x3 grid: (3-1)*3 + (3-1)*3 = 12 arcs
        assert model.num_cost == 12

    def test_setObj_and_solve(self, model):
        cost = np.random.RandomState(42).rand(12)
        model.setObj(cost)
        sol, obj = model.solve()
        assert len(sol) == model.num_cost
        assert isinstance(obj, float)
        # sanity: objective equals c.T sol
        np.testing.assert_allclose(obj, np.dot(cost, np.asarray(sol)), atol=1e-3)

    def test_setObj_wrong_size(self, model):
        with pytest.raises(ValueError):
            model.setObj(np.zeros(99))

    def test_copy(self, model):
        cost = np.random.RandomState(42).rand(12)
        model.setObj(cost)
        _, obj1 = model.solve()
        new_model = model.copy()
        new_model.setObj(cost)
        _, obj2 = new_model.solve()
        np.testing.assert_allclose(obj1, obj2, atol=1e-3)

    def test_addConstr(self, model):
        cost = np.random.RandomState(42).rand(12)
        model.setObj(cost)
        _, obj0 = model.solve()
        # adding redundant constraint sum(x) >= 0 should not change optimum much
        new_model = model.addConstr([1.0] * 12, 0.0)
        new_model.setObj(cost)
        _, obj1 = new_model.solve()
        # new constraint is non-binding, obj should be unchanged
        np.testing.assert_allclose(obj0, obj1, atol=1e-3)

    @requires_mpax_gurobi
    def test_addConstr_matches_gurobi(self, model):
        # PyEPO convention: addConstr enforces `coefs · x <= rhs`; verify mpax matches grb
        cost = np.random.RandomState(42).rand(12)
        mpax_constrained = model.addConstr([1.0] * 12, 5.0)
        mpax_constrained.setObj(cost)
        _, mpax_obj = mpax_constrained.solve()
        grb_constrained = shortestPathModel(grid=(3, 3)).addConstr(np.ones(12), 5.0)
        grb_constrained.setObj(cost)
        _, grb_obj = grb_constrained.solve()
        np.testing.assert_allclose(mpax_obj, grb_obj, atol=1e-2)


@requires_mpax
class TestMpaxKnapsackModel:

    @pytest.fixture
    def model(self):
        weights = np.array([[3.0, 4.0, 5.0, 6.0]])
        capacity = np.array([10.0])
        return mpaxKnapsackModel(weights=weights, capacity=capacity)

    def test_init(self, model):
        assert model.modelSense == EPO.MAXIMIZE

    def test_num_cost(self, model):
        assert model.num_cost == 4

    def test_solve(self, model):
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        model.setObj(cost)
        sol, obj = model.solve()
        assert len(sol) == model.num_cost
        assert isinstance(obj, float)
        # LP relaxation: items can be fractional, but all in [0, 1]
        sol_np = np.asarray(sol)
        assert sol_np.min() >= -1e-3
        assert sol_np.max() <= 1.0 + 1e-3

    def test_setObj_wrong_size(self, model):
        with pytest.raises(ValueError):
            model.setObj(np.zeros(99))


# ============================================================
# Cross-backend consistency: MPAX vs Gurobi
# ============================================================

@requires_mpax_gurobi
class TestCrossBackendMpax:

    def test_shortestpath_grb_vs_mpax(self):
        # Shortest path LP relaxation matches IP because constraint matrix is TU
        cost = np.random.RandomState(42).rand(12)
        grb_model = shortestPathModel(grid=(3, 3))
        grb_model.setObj(cost)
        _, grb_obj = grb_model.solve()
        mpax_model = mpaxShortestPathModel(grid=(3, 3))
        mpax_model.setObj(cost)
        _, mpax_obj = mpax_model.solve()
        np.testing.assert_allclose(grb_obj, mpax_obj, atol=1e-2)

    def test_knapsack_lp_grb_vs_mpax(self):
        # MPAX knapsack is LP relaxation — compare against grb knapsackModelRel
        weights = np.array([[3.0, 4.0, 5.0, 6.0]])
        capacity = np.array([10.0])
        cost = np.array([10.0, 6.0, 3.0, 2.0])
        grb_model = knapsackModelRel(weights=weights, capacity=capacity)
        grb_model.setObj(cost)
        _, grb_obj = grb_model.solve()
        mpax_model = mpaxKnapsackModel(weights=weights, capacity=capacity)
        mpax_model.setObj(cost)
        _, mpax_obj = mpax_model.solve()
        np.testing.assert_allclose(grb_obj, mpax_obj, atol=1e-2)


# ============================================================
# CVRP shared fixture data (5 nodes, 2 vehicles, capacity 5)
# ============================================================

_VRP_NUM_NODES = 5
_VRP_DEMANDS = [2.0, 1.0, 3.0, 2.0]
_VRP_CAPACITY = 5.0
_VRP_NUM_VEHICLES = 2
_VRP_NUM_EDGES = _VRP_NUM_NODES * (_VRP_NUM_NODES - 1) // 2
_VRP_COST = np.random.RandomState(0).rand(_VRP_NUM_EDGES)


# ============================================================
# CVRP: Gurobi RCI (lazy rounded-capacity cuts)
# ============================================================

@requires_gurobi
class TestVrpRCIModel:

    @pytest.fixture
    def model(self):
        return vrpRCIModel(
            num_nodes=_VRP_NUM_NODES, demands=_VRP_DEMANDS,
            capacity=_VRP_CAPACITY, num_vehicle=_VRP_NUM_VEHICLES,
        )

    def test_init(self, model):
        assert model.num_nodes == _VRP_NUM_NODES
        assert model.modelSense == EPO.MINIMIZE
        assert model.capacity == _VRP_CAPACITY

    def test_num_cost(self, model):
        assert model.num_cost == _VRP_NUM_EDGES

    def test_setObj_and_solve(self, model):
        model.setObj(_VRP_COST)
        sol, obj = model.solve()
        sol = np.array(sol)
        assert len(sol) == model.num_cost
        assert isinstance(obj, float)
        # vertex must be binary
        assert np.allclose(sol, np.round(sol), atol=1e-6)

    def test_lazy_constrs_tracked(self, model):
        # _lazy_constrs attribute exists after build, populated after solve if cuts fire
        model.setObj(_VRP_COST)
        model.solve()
        assert hasattr(model._model, "_lazy_constrs")
        assert isinstance(model._model._lazy_constrs, list)

    def test_getTour_depot_anchored(self, model):
        model.setObj(_VRP_COST)
        sol, _ = model.solve()
        tours = model.getTour(sol)
        # every tour must start and end at depot 0
        assert all(t[0] == 0 and t[-1] == 0 for t in tours)
        # all customers covered exactly once across tours
        visited = [v for t in tours for v in t[1:-1]]
        assert sorted(visited) == list(range(1, _VRP_NUM_NODES))

    def test_setObj_wrong_size(self, model):
        with pytest.raises(ValueError):
            model.setObj(np.ones(3))

    def test_copy(self, model):
        model.setObj(_VRP_COST)
        _, obj1 = model.solve()
        m2 = model.copy()
        m2.setObj(_VRP_COST)
        _, obj2 = m2.solve()
        np.testing.assert_allclose(obj1, obj2, atol=1e-6)


# ============================================================
# CVRP: Gurobi MTZ (directed, no lazy cuts)
# ============================================================

@requires_gurobi
class TestVrpMTZModel:

    @pytest.fixture
    def model(self):
        return vrpMTZModel(
            num_nodes=_VRP_NUM_NODES, demands=_VRP_DEMANDS,
            capacity=_VRP_CAPACITY, num_vehicle=_VRP_NUM_VEHICLES,
        )

    def test_init(self, model):
        assert model.num_nodes == _VRP_NUM_NODES
        assert model.modelSense == EPO.MINIMIZE

    def test_num_cost(self, model):
        # one cost per undirected edge (NOT per directed Var)
        assert model.num_cost == _VRP_NUM_EDGES

    def test_setObj_and_solve(self, model):
        model.setObj(_VRP_COST)
        sol, obj = model.solve()
        sol = np.array(sol)
        assert len(sol) == model.num_cost
        assert np.allclose(sol, np.round(sol), atol=1e-6)

    def test_relax_is_lp(self, model):
        rel = model.relax()
        assert isinstance(rel, vrpMTZModelRel)
        rel.setObj(_VRP_COST)
        _, obj_rel = rel.solve()
        model.setObj(_VRP_COST)
        _, obj_int = model.solve()
        # MINIMIZE: LP bound <= IP optimum
        assert obj_rel <= obj_int + 1e-6

    def test_relax_cannot_relax_again(self, model):
        rel = model.relax()
        with pytest.raises(RuntimeError):
            rel.relax()

    def test_relax_no_getTour(self, model):
        rel = model.relax()
        with pytest.raises(RuntimeError):
            rel.getTour([0] * model.num_cost)


# ============================================================
# CVRP: RCI vs MTZ formulation consistency (Gurobi)
# ============================================================

@requires_gurobi
class TestVrpCrossFormulation:

    def test_rci_vs_mtz_objective(self):
        rci = vrpRCIModel(
            num_nodes=_VRP_NUM_NODES, demands=_VRP_DEMANDS,
            capacity=_VRP_CAPACITY, num_vehicle=_VRP_NUM_VEHICLES,
        )
        rci.setObj(_VRP_COST)
        _, obj_rci = rci.solve()
        mtz = vrpMTZModel(
            num_nodes=_VRP_NUM_NODES, demands=_VRP_DEMANDS,
            capacity=_VRP_CAPACITY, num_vehicle=_VRP_NUM_VEHICLES,
        )
        mtz.setObj(_VRP_COST)
        _, obj_mtz = mtz.solve()
        np.testing.assert_allclose(obj_rci, obj_mtz, atol=1e-4)


# ============================================================
# CVRP: COPT RCI (lazy cuts via CallbackBase)
# ============================================================

@requires_copt
class TestCoptVrpRCIModel:

    @pytest.fixture
    def model(self):
        return coptVrpRCIModel(
            num_nodes=_VRP_NUM_NODES, demands=_VRP_DEMANDS,
            capacity=_VRP_CAPACITY, num_vehicle=_VRP_NUM_VEHICLES,
        )

    def test_num_cost(self, model):
        assert model.num_cost == _VRP_NUM_EDGES

    def test_setObj_and_solve(self, model):
        model.setObj(_VRP_COST)
        sol, obj = model.solve()
        sol = np.array(sol)
        assert np.allclose(sol, np.round(sol), atol=1e-6)

    def test_getTour_depot_anchored(self, model):
        model.setObj(_VRP_COST)
        sol, _ = model.solve()
        tours = model.getTour(sol)
        assert all(t[0] == 0 and t[-1] == 0 for t in tours)


# ============================================================
# CVRP: COPT MTZ
# ============================================================

@requires_copt
class TestCoptVrpMTZModel:

    @pytest.fixture
    def model(self):
        return coptVrpMTZModel(
            num_nodes=_VRP_NUM_NODES, demands=_VRP_DEMANDS,
            capacity=_VRP_CAPACITY, num_vehicle=_VRP_NUM_VEHICLES,
        )

    def test_setObj_and_solve(self, model):
        model.setObj(_VRP_COST)
        sol, obj = model.solve()
        sol = np.array(sol)
        assert np.allclose(sol, np.round(sol), atol=1e-6)

    def test_relax_is_lp(self, model):
        rel = model.relax()
        assert isinstance(rel, coptVrpMTZModelRel)
        rel.setObj(_VRP_COST)
        _, obj_rel = rel.solve()
        model.setObj(_VRP_COST)
        _, obj_int = model.solve()
        assert obj_rel <= obj_int + 1e-6


# ============================================================
# CVRP: Pyomo MTZ (no RCI: Pyomo lacks lazy callbacks)
# ============================================================

@requires_pyomo
class TestOmoVrpMTZModel:

    @pytest.fixture
    def model(self):
        return omoVrpMTZModel(
            num_nodes=_VRP_NUM_NODES, demands=_VRP_DEMANDS,
            capacity=_VRP_CAPACITY, num_vehicle=_VRP_NUM_VEHICLES,
            solver="gurobi",
        )

    def test_setObj_and_solve(self, model):
        model.setObj(_VRP_COST)
        sol, obj = model.solve()
        sol = np.array(sol)
        assert np.allclose(sol, np.round(sol), atol=1e-6)

    def test_relax_is_lp(self, model):
        rel = model.relax()
        assert isinstance(rel, omoVrpMTZModelRel)
        rel.setObj(_VRP_COST)
        _, obj_rel = rel.solve()
        model.setObj(_VRP_COST)
        _, obj_int = model.solve()
        assert obj_rel <= obj_int + 1e-6


# ============================================================
# CVRP: Cross-backend consistency (grb / copt / omo)
# ============================================================

@pytest.mark.skipif(not (_HAS_GUROBI and _HAS_COPT), reason="Gurobi or COPT not installed")
class TestVrpCrossBackendCopt:

    def test_mtz_grb_vs_copt(self):
        grb = vrpMTZModel(
            num_nodes=_VRP_NUM_NODES, demands=_VRP_DEMANDS,
            capacity=_VRP_CAPACITY, num_vehicle=_VRP_NUM_VEHICLES,
        )
        grb.setObj(_VRP_COST)
        _, obj_grb = grb.solve()
        copt = coptVrpMTZModel(
            num_nodes=_VRP_NUM_NODES, demands=_VRP_DEMANDS,
            capacity=_VRP_CAPACITY, num_vehicle=_VRP_NUM_VEHICLES,
        )
        copt.setObj(_VRP_COST)
        _, obj_copt = copt.solve()
        np.testing.assert_allclose(obj_grb, obj_copt, atol=1e-4)


@requires_pyomo
class TestVrpCrossBackendOmo:

    def test_mtz_grb_vs_omo(self):
        grb = vrpMTZModel(
            num_nodes=_VRP_NUM_NODES, demands=_VRP_DEMANDS,
            capacity=_VRP_CAPACITY, num_vehicle=_VRP_NUM_VEHICLES,
        )
        grb.setObj(_VRP_COST)
        _, obj_grb = grb.solve()
        omo = omoVrpMTZModel(
            num_nodes=_VRP_NUM_NODES, demands=_VRP_DEMANDS,
            capacity=_VRP_CAPACITY, num_vehicle=_VRP_NUM_VEHICLES,
            solver="gurobi",
        )
        omo.setObj(_VRP_COST)
        _, obj_omo = omo.solve()
        np.testing.assert_allclose(obj_grb, obj_omo, atol=1e-4)
