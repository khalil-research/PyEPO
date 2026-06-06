#!/usr/bin/env python
"""Tests for pyepo.model: optimization model wrappers.

The optModel contract (init / num_cost / setObj+solve / copy / relax / addConstr)
is identical across solver backends, so each problem is tested once and
parametrized over its supported backends. Backends that are not installed skip
via per-param marks rather than failing. Formulation-specific behaviour
(TSP GG/DFJ/MTZ, VRP RCI/MTZ, MPAX QP, CP-SAT integer guard) is tested
explicitly, plus cross-backend objective parity against Gurobi.
"""

import numpy as np
import pytest

from pyepo import EPO
from pyepo.model.opt import optModel

from .conftest import (
    _HAS_GUROBI,
    _HAS_MPAX,
    _HAS_PYOMO,
    requires_copt,
    requires_gurobi,
    requires_mpax,
    requires_ortools,
)

# omo models here always use solver="gurobi", so they need both backends
requires_omo = pytest.mark.skipif(
    not (_HAS_PYOMO and _HAS_GUROBI), reason="Pyomo or Gurobi not installed")


def _to_np(sol):
    """MPAX returns a torch.Tensor (possibly CUDA); normalize to CPU numpy."""
    if hasattr(sol, "cpu"):
        return sol.cpu().numpy()
    return np.asarray(sol)


# ============================================================
# optModel base class
# ============================================================

class TestOptModelBase:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            optModel()


# ============================================================
# Knapsack (MAXIMIZE, binary) — parametrized over backends
# ============================================================

_KNAP_W = np.array([[3.0, 4.0, 5.0, 6.0]])
_KNAP_CAP = np.array([10.0])
_KNAP_COST = np.array([10.0, 6.0, 3.0, 2.0])  # unique optimum [1, 1, 0, 0]


def _make_knapsack(backend):
    """Return (model, meta) for a knapsack on the given backend."""
    if backend == "grb":
        from pyepo.model.grb.knapsack import knapsackModel, knapsackModelRel
        m = knapsackModel(weights=_KNAP_W, capacity=_KNAP_CAP)
        return m, {"relax": knapsackModelRel, "binary": True, "tol": 1e-6}
    if backend == "copt":
        from pyepo.model.copt.knapsack import knapsackModel, knapsackModelRel
        m = knapsackModel(weights=_KNAP_W, capacity=_KNAP_CAP)
        return m, {"relax": knapsackModelRel, "binary": True, "tol": 1e-6}
    if backend == "omo":
        from pyepo.model.omo.knapsack import knapsackModel, knapsackModelRel
        m = knapsackModel(weights=_KNAP_W, capacity=_KNAP_CAP, solver="gurobi")
        return m, {"relax": knapsackModelRel, "binary": True, "tol": 1e-6}
    if backend == "ort":
        from pyepo.model.ort.knapsack import knapsackModel, knapsackModelRel
        m = knapsackModel(weights=_KNAP_W, capacity=_KNAP_CAP)
        return m, {"relax": knapsackModelRel, "binary": True, "tol": 1e-6}
    if backend == "ortcp":
        from pyepo.model.ort.knapsack import knapsackCpModel
        m = knapsackCpModel(weights=_KNAP_W.astype(int), capacity=_KNAP_CAP.astype(int))
        return m, {"relax": None, "binary": True, "tol": 1e-2}
    if backend == "mpax":
        from pyepo.model.mpax.knapsack import knapsackModel
        m = knapsackModel(weights=_KNAP_W, capacity=_KNAP_CAP)
        return m, {"relax": None, "binary": False, "tol": 1e-2}  # LP relaxation
    raise ValueError(backend)


_KNAP_BACKENDS = [
    pytest.param("grb", marks=requires_gurobi),
    pytest.param("copt", marks=requires_copt),
    pytest.param("omo", marks=requires_omo),
    pytest.param("ort", marks=requires_ortools),
    pytest.param("ortcp", marks=requires_ortools),
    pytest.param("mpax", marks=requires_mpax),
]


@pytest.mark.parametrize("backend", _KNAP_BACKENDS)
class TestKnapsack:

    def test_init_and_num_cost(self, backend):
        m, _ = _make_knapsack(backend)
        assert m.modelSense == EPO.MAXIMIZE
        assert m.num_cost == 4

    def test_setObj_and_solve(self, backend):
        m, meta = _make_knapsack(backend)
        m.setObj(_KNAP_COST)
        sol, obj = m.solve()
        sol = _to_np(sol)
        assert len(sol) == m.num_cost
        assert isinstance(obj, float)
        # feasible: weights @ sol <= capacity
        assert np.all(_KNAP_W @ sol <= _KNAP_CAP + 1e-3)
        np.testing.assert_allclose(obj, float(_KNAP_COST @ sol), atol=meta["tol"])
        if meta["binary"]:
            np.testing.assert_allclose(sol, np.round(sol), atol=1e-3)
        else:
            assert sol.min() >= -1e-3 and sol.max() <= 1.0 + 1e-3

    def test_setObj_wrong_size_raises(self, backend):
        m, _ = _make_knapsack(backend)
        with pytest.raises(ValueError):
            m.setObj(np.ones(10))

    def test_copy_isolation(self, backend):
        m, meta = _make_knapsack(backend)
        m.setObj(_KNAP_COST)
        _, obj1 = m.solve()
        m2 = m.copy()
        m2.setObj(_KNAP_COST)
        _, obj2 = m2.solve()
        np.testing.assert_allclose(obj1, obj2, atol=meta["tol"])

    def test_addConstr_no_improvement(self, backend):
        # MAXIMIZE: a tighter constraint cannot increase the objective
        m, meta = _make_knapsack(backend)
        m.setObj(_KNAP_COST)
        _, obj1 = m.solve()
        m2 = m.addConstr(np.ones(m.num_cost), 1)
        m2.setObj(_KNAP_COST)
        _, obj2 = m2.solve()
        assert obj2 <= obj1 + max(meta["tol"], 1e-6)

    def test_relax(self, backend):
        m, meta = _make_knapsack(backend)
        if meta["relax"] is None:
            pytest.skip("backend has no LP relaxation")
        rel = m.relax()
        assert isinstance(rel, meta["relax"])
        rel.setObj(_KNAP_COST)
        _, obj_rel = rel.solve()
        m.setObj(_KNAP_COST)
        _, obj_int = m.solve()
        # LP bound >= IP optimum for MAXIMIZE
        assert obj_rel >= obj_int - meta["tol"]
        with pytest.raises(RuntimeError):
            rel.relax()


def _make_knapsack_multidim(backend):
    """A 2-dimensional knapsack on the given backend (distinct code path)."""
    w = np.array([[3.0, 4.0, 5.0], [2.0, 3.0, 4.0]])
    cap = np.array([8.0, 7.0])
    if backend == "grb":
        from pyepo.model.grb.knapsack import knapsackModel
        return knapsackModel(weights=w, capacity=cap), w, cap
    if backend == "copt":
        from pyepo.model.copt.knapsack import knapsackModel
        return knapsackModel(weights=w, capacity=cap), w, cap
    if backend == "omo":
        from pyepo.model.omo.knapsack import knapsackModel
        return knapsackModel(weights=w, capacity=cap, solver="gurobi"), w, cap
    if backend == "ort":
        from pyepo.model.ort.knapsack import knapsackModel
        return knapsackModel(weights=w, capacity=cap), w, cap
    raise ValueError(backend)


@pytest.mark.parametrize("backend", [
    pytest.param("grb", marks=requires_gurobi),
    pytest.param("copt", marks=requires_copt),
    pytest.param("omo", marks=requires_omo),
    pytest.param("ort", marks=requires_ortools),
])
def test_knapsack_multidimensional(backend):
    m, w, cap = _make_knapsack_multidim(backend)
    assert m.num_cost == 3
    m.setObj(np.array([5.0, 4.0, 3.0]))
    sol, _ = m.solve()
    sol = _to_np(sol)
    # feasible against both dimensions
    assert np.all(w @ sol <= cap + 1e-6)


# ============================================================
# Shortest path (MINIMIZE, LP) — parametrized over backends
# ============================================================

def _make_shortestpath(backend, grid=(3, 3)):
    if backend == "grb":
        from pyepo.model.grb.shortestpath import shortestPathModel
        return shortestPathModel(grid=grid), {"integral": True, "tol": 1e-6}
    if backend == "copt":
        from pyepo.model.copt.shortestpath import shortestPathModel
        return shortestPathModel(grid=grid), {"integral": True, "tol": 1e-6}
    if backend == "omo":
        from pyepo.model.omo.shortestpath import shortestPathModel
        return shortestPathModel(grid=grid, solver="gurobi"), {"integral": True, "tol": 1e-6}
    if backend == "ort":
        from pyepo.model.ort.shortestpath import shortestPathModel
        return shortestPathModel(grid=grid), {"integral": True, "tol": 1e-6}
    if backend == "ortcp":
        from pyepo.model.ort.shortestpath import shortestPathCpModel
        return shortestPathCpModel(grid=grid), {"integral": True, "tol": 1e-2}
    if backend == "mpax":
        from pyepo.model.mpax.shortestpath import shortestPathModel
        return shortestPathModel(grid=grid), {"integral": False, "tol": 1e-2}
    raise ValueError(backend)


_SP_BACKENDS = [
    pytest.param("grb", marks=requires_gurobi),
    pytest.param("copt", marks=requires_copt),
    pytest.param("omo", marks=requires_omo),
    pytest.param("ort", marks=requires_ortools),
    pytest.param("ortcp", marks=requires_ortools),
    pytest.param("mpax", marks=requires_mpax),
]


@pytest.mark.parametrize("backend", _SP_BACKENDS)
class TestShortestPath:

    def test_init_and_num_cost(self, backend):
        m, _ = _make_shortestpath(backend)
        assert m.modelSense == EPO.MINIMIZE
        # 3x3 grid: 6 horizontal + 6 vertical edges
        assert m.num_cost == 12

    def test_setObj_and_solve(self, backend):
        m, meta = _make_shortestpath(backend)
        cost = np.random.RandomState(42).rand(m.num_cost)
        m.setObj(cost)
        sol, obj = m.solve()
        sol = _to_np(sol)
        assert len(sol) == m.num_cost
        assert isinstance(obj, float)
        assert obj > 0
        assert sol.min() >= -1e-3 and sol.max() <= 1.0 + 1e-3
        if meta["integral"]:
            np.testing.assert_allclose(sol, np.round(sol), atol=1e-6)

    def test_setObj_wrong_size_raises(self, backend):
        m, _ = _make_shortestpath(backend)
        with pytest.raises(ValueError):
            m.setObj(np.ones(5))

    def test_copy_isolation(self, backend):
        m, meta = _make_shortestpath(backend)
        cost = np.random.RandomState(42).rand(m.num_cost)
        m.setObj(cost)
        _, obj1 = m.solve()
        m2 = m.copy()
        m2.setObj(cost)
        _, obj2 = m2.solve()
        np.testing.assert_allclose(obj1, obj2, atol=meta["tol"])

    def test_addConstr_no_improvement(self, backend):
        # MINIMIZE: a tighter constraint cannot decrease the objective
        m, meta = _make_shortestpath(backend)
        cost = np.random.RandomState(42).rand(m.num_cost)
        m2 = m.addConstr(np.ones(m.num_cost), 5)
        m2.setObj(cost)
        _, obj2 = m2.solve()
        m.setObj(cost)
        _, obj1 = m.solve()
        assert obj2 >= obj1 - max(meta["tol"], 1e-6)

    def test_grid_sizes_num_cost(self, backend):
        for grid in [(2, 2), (3, 4), (5, 5)]:
            m, _ = _make_shortestpath(backend, grid=grid)
            assert m.num_cost == (grid[0] - 1) * grid[1] + (grid[1] - 1) * grid[0]


# ============================================================
# Portfolio (MAXIMIZE, QP) — parametrized over backends
# ============================================================

def _portfolio_data(num_assets=10):
    from pyepo.data.portfolio import genData
    cov, _, revenue = genData(num_data=10, num_features=4, num_assets=num_assets, deg=1)
    return cov, revenue


def _make_portfolio(backend, cov, num_assets=10):
    if backend == "grb":
        from pyepo.model.grb.portfolio import portfolioModel
        return portfolioModel(num_assets=num_assets, covariance=cov)
    if backend == "copt":
        from pyepo.model.copt.portfolio import portfolioModel
        return portfolioModel(num_assets=num_assets, covariance=cov)
    if backend == "omo":
        from pyepo.model.omo.portfolio import portfolioModel
        return portfolioModel(num_assets=num_assets, covariance=cov, solver="gurobi")
    raise ValueError(backend)


_PORTFOLIO_BACKENDS = [
    pytest.param("grb", marks=requires_gurobi),
    pytest.param("copt", marks=requires_copt),
    pytest.param("omo", marks=requires_omo),
]


@pytest.mark.parametrize("backend", _PORTFOLIO_BACKENDS)
class TestPortfolio:

    def test_init_and_num_cost(self, backend):
        cov, _ = _portfolio_data()
        m = _make_portfolio(backend, cov)
        assert m.modelSense == EPO.MAXIMIZE
        assert m.num_cost == 10

    def test_solve_budget_constraint(self, backend):
        cov, revenue = _portfolio_data()
        m = _make_portfolio(backend, cov)
        m.setObj(revenue[0])
        sol, obj = m.solve()
        sol = _to_np(sol)
        assert len(sol) == m.num_cost
        assert isinstance(obj, float)
        # budget: weights sum to 1
        np.testing.assert_allclose(np.sum(sol), 1.0, atol=1e-4)

    def test_copy_isolation(self, backend):
        cov, revenue = _portfolio_data()
        m = _make_portfolio(backend, cov)
        m.setObj(revenue[0])
        _, obj1 = m.solve()
        m2 = m.copy()
        m2.setObj(revenue[0])
        _, obj2 = m2.solve()
        np.testing.assert_allclose(obj1, obj2, atol=1e-6)

    def test_addConstr_no_improvement(self, backend):
        cov, revenue = _portfolio_data()
        m = _make_portfolio(backend, cov)
        m.setObj(revenue[0])
        _, obj1 = m.solve()
        coefs = np.zeros(m.num_cost)
        coefs[:3] = 1.0
        m2 = m.addConstr(coefs, 0.5)
        m2.setObj(revenue[0])
        _, obj2 = m2.solve()
        assert obj2 <= obj1 + 1e-6


# ============================================================
# TSP (MINIMIZE, binary) — parametrized over (backend, formulation)
# ============================================================

def _make_tsp(backend, formulation, num_nodes=4):
    """Return (model, relax_cls_or_None)."""
    if backend == "grb":
        from pyepo.model.grb import tsp as t
        cls = {"GG": t.tspGGModel, "DFJ": t.tspDFJModel, "MTZ": t.tspMTZModel}[formulation]
        rel = {"GG": t.tspGGModelRel, "DFJ": None, "MTZ": t.tspMTZModelRel}[formulation]
        m = cls(num_nodes=num_nodes)
        m._model.Params.OutputFlag = 0
        return m, rel
    if backend == "copt":
        from pyepo.model.copt import tsp as t
        cls = {"GG": t.tspGGModel, "DFJ": t.tspDFJModel, "MTZ": t.tspMTZModel}[formulation]
        rel = {"GG": t.tspGGModelRel, "DFJ": None, "MTZ": t.tspMTZModelRel}[formulation]
        return cls(num_nodes=num_nodes), rel
    if backend == "omo":
        from pyepo.model.omo import tsp as t
        cls = {"GG": t.tspGGModel, "MTZ": t.tspMTZModel}[formulation]
        rel = {"GG": t.tspGGModelRel, "MTZ": t.tspMTZModelRel}[formulation]
        return cls(num_nodes=num_nodes, solver="gurobi"), rel
    raise ValueError(backend)


_TSP_PARAMS = [
    pytest.param("grb", "GG", marks=requires_gurobi),
    pytest.param("grb", "DFJ", marks=requires_gurobi),
    pytest.param("grb", "MTZ", marks=requires_gurobi),
    pytest.param("copt", "GG", marks=requires_copt),
    pytest.param("copt", "DFJ", marks=requires_copt),
    pytest.param("copt", "MTZ", marks=requires_copt),
    pytest.param("omo", "GG", marks=requires_omo),
    pytest.param("omo", "MTZ", marks=requires_omo),
]


@pytest.mark.parametrize("backend,formulation", _TSP_PARAMS)
class TestTSP:

    def test_init_and_num_cost(self, backend, formulation):
        m, _ = _make_tsp(backend, formulation)
        assert m.num_nodes == 4
        assert m.modelSense == EPO.MINIMIZE
        assert m.num_cost == 6  # C(4, 2)
        assert len(m.edges) == 6

    def test_setObj_solve_is_tour(self, backend, formulation):
        m, _ = _make_tsp(backend, formulation)
        cost = np.random.RandomState(42).rand(m.num_cost)
        m.setObj(cost)
        sol, obj = m.solve()
        sol = np.asarray(sol)
        assert len(sol) == m.num_cost
        assert isinstance(obj, float)
        np.testing.assert_allclose(sol, np.round(sol), atol=1e-6)
        # a 4-node tour uses exactly 4 edges
        assert int(np.sum(sol)) == m.num_nodes
        tour = m.getTour(sol)
        assert tour[0] == tour[-1]
        assert len(set(tour[:-1])) == m.num_nodes

    def test_setObj_wrong_size_raises(self, backend, formulation):
        m, _ = _make_tsp(backend, formulation)
        with pytest.raises(ValueError):
            m.setObj(np.ones(3))

    def test_copy_runs(self, backend, formulation):
        m, _ = _make_tsp(backend, formulation)
        cost = np.random.RandomState(42).rand(m.num_cost)
        m2 = m.copy()
        m2.setObj(cost)
        _, obj = m2.solve()
        assert isinstance(obj, float)

    def test_addConstr_infeasible(self, backend, formulation):
        # 4-node tour needs 4 edges; sum(edges) <= 3 is infeasible
        m, _ = _make_tsp(backend, formulation)
        m2 = m.addConstr(np.ones(m.num_cost), 3)
        m2.setObj(np.random.RandomState(42).rand(m.num_cost))
        with pytest.raises(Exception):
            m2.solve()

    def test_relax(self, backend, formulation):
        m, rel_cls = _make_tsp(backend, formulation)
        if rel_cls is None:
            pytest.skip("formulation has no relaxation (lazy-cut DFJ)")
        rel = m.relax()
        assert isinstance(rel, rel_cls)
        cost = np.random.RandomState(42).rand(m.num_cost)
        rel.setObj(cost)
        _, obj_rel = rel.solve()
        m.setObj(cost)
        _, obj_int = m.solve()
        # MINIMIZE: LP bound <= IP optimum
        assert obj_rel <= obj_int + 1e-6
        with pytest.raises(RuntimeError):
            rel.relax()
        with pytest.raises(RuntimeError):
            rel.getTour([0] * m.num_cost)


_TSP_CONSISTENCY = [
    pytest.param("grb", marks=requires_gurobi),
    pytest.param("copt", marks=requires_copt),
]


@pytest.mark.parametrize("backend", _TSP_CONSISTENCY)
def test_tsp_formulations_agree(backend):
    cost = np.random.RandomState(42).rand(6)
    objs = []
    for formulation in ("GG", "DFJ", "MTZ"):
        m, _ = _make_tsp(backend, formulation)
        m.setObj(cost)
        _, obj = m.solve()
        objs.append(obj)
    np.testing.assert_allclose(objs[1], objs[0], atol=1e-4)
    np.testing.assert_allclose(objs[2], objs[0], atol=1e-4)


# ============================================================
# CVRP (MINIMIZE, binary) — parametrized over (backend, formulation)
# ============================================================

_VRP_NUM_NODES = 5
_VRP_DEMANDS = [2.0, 1.0, 3.0, 2.0]
_VRP_CAPACITY = 5.0
_VRP_NUM_VEHICLES = 2
_VRP_NUM_EDGES = _VRP_NUM_NODES * (_VRP_NUM_NODES - 1) // 2
_VRP_COST = np.random.RandomState(0).rand(_VRP_NUM_EDGES)


def _make_vrp(backend, formulation):
    """Return (model, relax_cls_or_None)."""
    kw = {"num_nodes": _VRP_NUM_NODES, "demands": _VRP_DEMANDS,
          "capacity": _VRP_CAPACITY, "num_vehicle": _VRP_NUM_VEHICLES}
    if backend == "grb":
        from pyepo.model.grb import vrp as v
        if formulation == "RCI":
            return v.vrpRCIModel(**kw), None
        return v.vrpMTZModel(**kw), v.vrpMTZModelRel
    if backend == "copt":
        from pyepo.model.copt import vrp as v
        if formulation == "RCI":
            return v.vrpRCIModel(**kw), None
        return v.vrpMTZModel(**kw), v.vrpMTZModelRel
    if backend == "omo":
        from pyepo.model.omo import vrp as v
        return v.vrpMTZModel(solver="gurobi", **kw), v.vrpMTZModelRel
    raise ValueError(backend)


_VRP_PARAMS = [
    pytest.param("grb", "RCI", marks=requires_gurobi),
    pytest.param("grb", "MTZ", marks=requires_gurobi),
    pytest.param("copt", "RCI", marks=requires_copt),
    pytest.param("copt", "MTZ", marks=requires_copt),
    pytest.param("omo", "MTZ", marks=requires_omo),
]


@pytest.mark.parametrize("backend,formulation", _VRP_PARAMS)
class TestVRP:

    def test_init_and_num_cost(self, backend, formulation):
        m, _ = _make_vrp(backend, formulation)
        assert m.num_nodes == _VRP_NUM_NODES
        assert m.modelSense == EPO.MINIMIZE
        assert m.num_cost == _VRP_NUM_EDGES

    def test_setObj_solve_binary(self, backend, formulation):
        m, _ = _make_vrp(backend, formulation)
        m.setObj(_VRP_COST)
        sol, obj = m.solve()
        sol = np.asarray(sol)
        assert len(sol) == m.num_cost
        assert isinstance(obj, float)
        np.testing.assert_allclose(sol, np.round(sol), atol=1e-6)

    def test_getTour_depot_anchored(self, backend, formulation):
        m, _ = _make_vrp(backend, formulation)
        m.setObj(_VRP_COST)
        sol, _ = m.solve()
        tours = m.getTour(sol)
        assert all(t[0] == 0 and t[-1] == 0 for t in tours)
        visited = sorted(v for t in tours for v in t[1:-1])
        assert visited == list(range(1, _VRP_NUM_NODES))

    def test_relax(self, backend, formulation):
        m, rel_cls = _make_vrp(backend, formulation)
        if rel_cls is None:
            pytest.skip("RCI has no relaxation (lazy cuts)")
        rel = m.relax()
        assert isinstance(rel, rel_cls)
        rel.setObj(_VRP_COST)
        _, obj_rel = rel.solve()
        m.setObj(_VRP_COST)
        _, obj_int = m.solve()
        assert obj_rel <= obj_int + 1e-6
        with pytest.raises(RuntimeError):
            rel.getTour([0] * m.num_cost)


@requires_gurobi
def test_vrp_formulations_agree():
    rci, _ = _make_vrp("grb", "RCI")
    rci.setObj(_VRP_COST)
    _, obj_rci = rci.solve()
    mtz, _ = _make_vrp("grb", "MTZ")
    mtz.setObj(_VRP_COST)
    _, obj_mtz = mtz.solve()
    np.testing.assert_allclose(obj_rci, obj_mtz, atol=1e-4)


@requires_gurobi
def test_vrp_rci_lazy_constrs_tracked():
    m, _ = _make_vrp("grb", "RCI")
    m.setObj(_VRP_COST)
    m.solve()
    assert isinstance(m._model._lazy_constrs, list)


# ============================================================
# Cross-backend objective parity (vs Gurobi reference)
# ============================================================

@requires_gurobi
@pytest.mark.parametrize("backend", [
    pytest.param("copt", marks=requires_copt),
    pytest.param("omo", marks=requires_omo),
    pytest.param("ort", marks=requires_ortools),
])
def test_knapsack_parity_vs_gurobi(backend):
    ref, _ = _make_knapsack("grb")
    ref.setObj(_KNAP_COST)
    _, ref_obj = ref.solve()
    m, _ = _make_knapsack(backend)
    m.setObj(_KNAP_COST)
    _, obj = m.solve()
    np.testing.assert_allclose(obj, ref_obj, atol=1e-4)


@requires_gurobi
@pytest.mark.parametrize("backend", [
    pytest.param("copt", marks=requires_copt),
    pytest.param("omo", marks=requires_omo),
    pytest.param("ort", marks=requires_ortools),
    pytest.param("mpax", marks=requires_mpax),  # SP LP relax == IP (TU matrix)
])
def test_shortestpath_parity_vs_gurobi(backend):
    cost = np.random.RandomState(42).rand(12)
    ref, _ = _make_shortestpath("grb")
    ref.setObj(cost)
    _, ref_obj = ref.solve()
    m, _meta = _make_shortestpath(backend)
    m.setObj(cost)
    _, obj = m.solve()
    # exact LP solvers match Gurobi tightly; MPAX is first-order PDHG (~1e-3)
    atol = 1e-2 if backend == "mpax" else 1e-4
    np.testing.assert_allclose(obj, ref_obj, atol=atol)


@requires_gurobi
@pytest.mark.parametrize("backend", [
    pytest.param("copt", marks=requires_copt),
    pytest.param("omo", marks=requires_omo),
])
def test_portfolio_parity_vs_gurobi(backend):
    cov, revenue = _portfolio_data()
    ref = _make_portfolio("grb", cov)
    ref.setObj(revenue[0])
    _, ref_obj = ref.solve()
    m = _make_portfolio(backend, cov)
    m.setObj(revenue[0])
    _, obj = m.solve()
    np.testing.assert_allclose(obj, ref_obj, atol=1e-4)


# ============================================================
# OR-Tools CP-SAT integer guard
# ============================================================

@requires_ortools
class TestOrtCpSatGuards:

    def test_float_weights_raise(self):
        from pyepo.model.ort.knapsack import knapsackCpModel
        with pytest.raises(ValueError, match="integer weights"):
            knapsackCpModel(weights=np.array([[3.5, 4.0, 5.0, 6.0]]), capacity=np.array([10]))

    def test_float_capacity_raise(self):
        from pyepo.model.ort.knapsack import knapsackCpModel
        with pytest.raises(ValueError, match="integer capacity"):
            knapsackCpModel(weights=np.array([[3, 4, 5, 6]]), capacity=np.array([10.5]))

    def test_relax_not_supported(self):
        from pyepo.model.ort.knapsack import knapsackCpModel
        m = knapsackCpModel(weights=np.array([[3, 4, 5, 6]]), capacity=np.array([10]))
        with pytest.raises(RuntimeError):
            m.relax()


# ============================================================
# MPAX QP path: optMpaxModel.Q routes to create_qp via raPDHG
# ============================================================

if _HAS_MPAX:
    import jax.numpy as jnp

    from pyepo.model.mpax.mpaxmodel import optMpaxModel

    class _MpaxBoxQP(optMpaxModel):
        """min 0.5 xᵀQx + cᵀx s.t. sum(x) >= -1000, l <= x <= u.

        Diagonal Q; the slack inequality is non-binding and exists only to give
        PDHG a non-empty dual block. Unconstrained optimum x*_i = -c_i / Q_ii.
        """
        use_sparse_matrix = False

        def __init__(self, Q_diag, lb=-10.0, ub=10.0):
            self._Q_diag = np.asarray(Q_diag, dtype=np.float32)
            self._lb, self._ub = float(lb), float(ub)
            super().__init__()

        def _getModel(self):
            n = self._Q_diag.shape[0]
            self.A = jnp.zeros((0, n), dtype=jnp.float32)
            self.b = jnp.zeros((0,), dtype=jnp.float32)
            self.G = jnp.ones((1, n), dtype=jnp.float32)
            self.h = jnp.array([-1000.0], dtype=jnp.float32)
            self.l = jnp.full(n, self._lb, dtype=jnp.float32)
            self.u = jnp.full(n, self._ub, dtype=jnp.float32)
            self.Q = jnp.diag(jnp.asarray(self._Q_diag, dtype=jnp.float32))
            return None, []


@requires_mpax
class TestMpaxQP:

    @pytest.fixture
    def model(self):
        return _MpaxBoxQP(Q_diag=[2.0, 4.0, 6.0, 8.0])

    def test_init_uses_qp_path(self, model):
        assert model.Q is not None
        assert model.Q.shape == (4, 4)
        assert model.modelSense == EPO.MINIMIZE

    def test_qp_closed_form(self, model):
        # c = -Q_diag => x* = [1,1,1,1], obj* = 0.5·sum(Q) - sum(Q) = -10
        model.setObj(np.array([-2.0, -4.0, -6.0, -8.0]))
        sol, obj = model.solve()
        np.testing.assert_allclose(_to_np(sol), [1.0, 1.0, 1.0, 1.0], atol=1e-2)
        np.testing.assert_allclose(obj, -10.0, atol=1e-2)

    def test_qp_batch_optimize(self, model):
        C = jnp.array([[-2.0, -4.0, -6.0, -8.0], [-1.0, -2.0, -3.0, -4.0]], dtype=jnp.float32)
        X, _objs = model.batch_optimize(C)
        np.testing.assert_allclose(_to_np(X[0]), [1.0, 1.0, 1.0, 1.0], atol=1e-2)
        np.testing.assert_allclose(_to_np(X[1]), [0.5, 0.5, 0.5, 0.5], atol=1e-2)

    def test_qp_addConstr_preserves_Q(self, model):
        model.setObj(np.array([-2.0, -4.0, -6.0, -8.0]))
        _, obj0 = model.solve()
        m2 = model.addConstr([1.0, 1.0, 1.0, 1.0], 2.0)  # binding (sum was 4)
        assert m2.Q is not None
        m2.setObj(np.array([-2.0, -4.0, -6.0, -8.0]))
        sol, obj1 = m2.solve()
        # PyEPO convention is coefs·x <= rhs: the solution must respect it.
        # (Replaces the old flaky exact-match-vs-Gurobi assertion; checking the
        # constraint directly is robust to PDHG's first-order tolerance.)
        assert np.sum(_to_np(sol)) <= 2.0 + 1e-2
        assert obj1 > obj0 - 1e-3  # MINIMIZE: binding constraint worsens objective

    def test_qp_rejects_maximize(self):
        class _BadMaxQP(_MpaxBoxQP):
            modelSense = EPO.MAXIMIZE
        with pytest.raises(ValueError, match="MINIMIZE"):
            _BadMaxQP(Q_diag=[1.0, 1.0])

    def test_lp_path_has_no_Q(self):
        m, _ = _make_shortestpath("mpax")
        assert m.Q is None

    def test_qp_lp_isolation(self, model):
        # a QP solve must not corrupt an independent LP model's path
        model.setObj(np.array([-2.0, -4.0, -6.0, -8.0]))
        model.solve()
        sp, _ = _make_shortestpath("mpax")
        sp.setObj(np.random.RandomState(0).rand(12))
        sol, obj = sp.solve()
        assert isinstance(obj, float)
        assert len(sol) == sp.num_cost
