#!/usr/bin/env python
# coding: utf-8
"""
Tests for pyepo.func: utility functions and abstract module

Focuses on unit-level tests (solution pool, caching, helpers) without training.
"""

import pytest
import numpy as np
import torch

from pyepo import EPO
from pyepo.func.utils import _cache_in_pass, _check_sol, _update_solution_pool, sumGammaDistribution

try:
    from pyepo.model.grb.shortestpath import shortestPathModel
    from pyepo.model.grb.knapsack import knapsackModel
    # verify Gurobi actually works (import may succeed without a valid license)
    shortestPathModel(grid=(3, 3))
    _HAS_GUROBI = True
except (ImportError, NameError, Exception):
    _HAS_GUROBI = False

try:
    from pyepo.model.ort.shortestpath import shortestPathModel as ortSPModel
    from pyepo.model.ort.knapsack import knapsackModel as ortKnapsackModel
    ortSPModel(grid=(3, 3))
    _HAS_ORTOOLS = True
except (ImportError, NameError, Exception):
    _HAS_ORTOOLS = False

requires_gurobi = pytest.mark.skipif(not _HAS_GUROBI, reason="Gurobi not installed")
requires_ortools = pytest.mark.skipif(not _HAS_ORTOOLS, reason="OR-Tools not installed")


# ============================================================
# _cache_in_pass tests
# ============================================================

class TestCacheInPass:

    def _make_minimize_model(self):
        from unittest.mock import MagicMock
        m = MagicMock()
        m.modelSense = EPO.MINIMIZE
        return m

    def _make_maximize_model(self):
        from unittest.mock import MagicMock
        m = MagicMock()
        m.modelSense = EPO.MAXIMIZE
        return m

    def test_minimize_selects_min_obj(self):
        model = self._make_minimize_model()
        # 2 instances, 3-dim cost
        cp = torch.tensor([[1.0, 2.0, 3.0],
                           [3.0, 2.0, 1.0]])
        # solution pool: 2 solutions
        solpool = torch.tensor([[1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0]])
        sol, obj, _ = _cache_in_pass(cp, model, solpool)
        # instance 0: costs=[1,2,3], sol0 gives obj=1, sol1 gives obj=3 → pick sol0
        assert torch.allclose(sol[0], solpool[0])
        # instance 1: costs=[3,2,1], sol0 gives obj=3, sol1 gives obj=1 → pick sol1
        assert torch.allclose(sol[1], solpool[1])

    def test_maximize_selects_max_obj(self):
        model = self._make_maximize_model()
        cp = torch.tensor([[1.0, 2.0, 3.0],
                           [3.0, 2.0, 1.0]])
        solpool = torch.tensor([[1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0]])
        sol, obj, _ = _cache_in_pass(cp, model, solpool)
        # instance 0: sol0→1, sol1→3 → pick sol1 (max)
        assert torch.allclose(sol[0], solpool[1])
        # instance 1: sol0→3, sol1→1 → pick sol0 (max)
        assert torch.allclose(sol[1], solpool[0])

    def test_invalid_sense_raises(self):
        from unittest.mock import MagicMock
        model = MagicMock()
        model.modelSense = 999
        cp = torch.tensor([[1.0, 2.0]])
        solpool = torch.tensor([[1.0, 0.0]])
        with pytest.raises(ValueError):
            _cache_in_pass(cp, model, solpool)

    def test_output_shapes(self):
        model = self._make_minimize_model()
        n, d, pool_size = 5, 4, 10
        cp = torch.randn(n, d)
        solpool = torch.randn(pool_size, d)
        sol, obj, _ = _cache_in_pass(cp, model, solpool)
        assert sol.shape == (n, d)
        assert obj.shape == (n,)


# ============================================================
# _check_sol tests
# ============================================================

class TestCheckSol:

    def test_correct_solution_passes(self):
        c = torch.tensor([[1.0, 2.0, 3.0]])
        w = torch.tensor([[1.0, 0.0, 1.0]])
        z = torch.tensor([4.0])  # 1*1 + 2*0 + 3*1 = 4
        # should not raise
        _check_sol(c, w, z)

    def test_incorrect_solution_raises(self):
        c = torch.tensor([[1.0, 2.0, 3.0]])
        w = torch.tensor([[1.0, 0.0, 1.0]])
        z = torch.tensor([999.0])  # wrong
        with pytest.raises(AssertionError):
            _check_sol(c, w, z)

    def test_batch(self):
        c = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        w = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        z = torch.tensor([3.0, 7.0])
        _check_sol(c, w, z)


# ============================================================
# sumGammaDistribution tests
# ============================================================

class TestSumGammaDistribution:

    def test_sample_shape(self):
        dist = sumGammaDistribution(kappa=1.0, n_iterations=10, seed=42)
        samples = dist.sample((5, 3))
        assert samples.shape == (5, 3)

    def test_deterministic(self):
        d1 = sumGammaDistribution(kappa=1.0, seed=42)
        d2 = sumGammaDistribution(kappa=1.0, seed=42)
        s1 = d1.sample((10,))
        s2 = d2.sample((10,))
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds(self):
        d1 = sumGammaDistribution(kappa=1.0, seed=0)
        d2 = sumGammaDistribution(kappa=1.0, seed=1)
        s1 = d1.sample((100,))
        s2 = d2.sample((100,))
        assert not np.array_equal(s1, s2)

    def test_finite_values(self):
        dist = sumGammaDistribution(kappa=2.0, n_iterations=20, seed=42)
        samples = dist.sample((100,))
        assert np.all(np.isfinite(samples))


# ============================================================
# Solution pool (_update_solution_pool) tests
# ============================================================

class TestSolutionPool:

    def test_init_pool(self):
        solset = set()
        sol = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        solpool = _update_solution_pool(sol, None, solset)
        assert solpool.shape == (2, 2)
        assert len(solset) == 2

    def test_dedup(self):
        solset = set()
        sol1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        solpool = _update_solution_pool(sol1, None, solset)
        # add same solutions again
        sol2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        solpool = _update_solution_pool(sol2, solpool, solset)
        # pool should still have only 2 solutions
        assert solpool.shape[0] == 2
        assert len(solset) == 2

    def test_add_new(self):
        solset = set()
        sol1 = torch.tensor([[1.0, 0.0]])
        solpool = _update_solution_pool(sol1, None, solset)
        sol2 = torch.tensor([[0.0, 1.0]])
        solpool = _update_solution_pool(sol2, solpool, solset)
        assert solpool.shape[0] == 2

    def test_mixed_new_and_dup(self):
        solset = set()
        sol1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        solpool = _update_solution_pool(sol1, None, solset)
        # one new, one duplicate
        sol2 = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        solpool = _update_solution_pool(sol2, solpool, solset)
        assert solpool.shape[0] == 3
        assert len(solset) == 3

    def test_empty_new_after_dedup(self):
        solset = set()
        sol = torch.tensor([[1.0, 2.0]])
        solpool = _update_solution_pool(sol, None, solset)
        # all duplicates
        solpool = _update_solution_pool(sol, solpool, solset)
        solpool = _update_solution_pool(sol, solpool, solset)
        assert solpool.shape[0] == 1


# ============================================================
# optModule init validation tests
# ============================================================

@requires_gurobi
class TestOptModuleInit:

    def test_invalid_model_type_raises(self):
        from pyepo.func.surrogate import SPOPlus
        with pytest.raises(TypeError):
            SPOPlus("not_a_model")

    def test_invalid_processes_raises(self):
        from pyepo.func.surrogate import SPOPlus
        model = shortestPathModel(grid=(3, 3))
        with pytest.raises(ValueError):
            SPOPlus(model, processes=-1)

    def test_invalid_solve_ratio_raises(self):
        from pyepo.func.surrogate import SPOPlus
        model = shortestPathModel(grid=(3, 3))
        with pytest.raises(ValueError):
            SPOPlus(model, solve_ratio=1.5)
        with pytest.raises(ValueError):
            SPOPlus(model, solve_ratio=-0.1)

    def test_solve_ratio_lt1_requires_dataset(self):
        from pyepo.func.surrogate import SPOPlus
        model = shortestPathModel(grid=(3, 3))
        with pytest.raises(TypeError):
            SPOPlus(model, solve_ratio=0.5, dataset=None)


# ============================================================
# dysOpt (DYS-Net) unit tests
# ============================================================

def _sp_constraint_matrices(grid):
    """Build flow conservation constraint matrices for shortest path."""
    from pyepo.model.opt import _get_grid_arcs
    arcs = _get_grid_arcs(grid)
    num_nodes = grid[0] * grid[1]
    num_arcs = len(arcs)
    A = np.zeros((num_nodes, num_arcs), dtype=np.float32)
    for idx, (s, e) in enumerate(arcs):
        A[s, idx] = 1
        A[e, idx] = -1
    b = np.zeros(num_nodes, dtype=np.float32)
    b[0] = 1
    b[num_nodes - 1] = -1
    l = np.zeros(num_arcs, dtype=np.float32)
    u = np.ones(num_arcs, dtype=np.float32)
    return A, b, l, u


class TestDysOptInit:

    def test_invalid_A_shape_raises(self):
        from pyepo.func.unrolling import dysOpt
        with pytest.raises(ValueError):
            dysOpt(np.ones(5, dtype=np.float32), np.ones(5, dtype=np.float32),
                   np.zeros(5, dtype=np.float32), np.ones(5, dtype=np.float32))

    def test_mismatched_b_length_raises(self):
        from pyepo.func.unrolling import dysOpt
        A = np.eye(3, 5, dtype=np.float32)
        b = np.ones(5, dtype=np.float32)  # wrong length (should be 3)
        l = np.zeros(5, dtype=np.float32)
        u = np.ones(5, dtype=np.float32)
        with pytest.raises(ValueError):
            dysOpt(A, b, l, u)

    def test_mismatched_lu_length_raises(self):
        from pyepo.func.unrolling import dysOpt
        A = np.eye(3, 5, dtype=np.float32)
        b = np.ones(3, dtype=np.float32)
        l = np.zeros(4, dtype=np.float32)  # wrong length (should be 5)
        u = np.ones(5, dtype=np.float32)
        with pytest.raises(ValueError):
            dysOpt(A, b, l, u)

    def test_invalid_alpha_raises(self):
        from pyepo.func.unrolling import dysOpt
        A, b, l, u = _sp_constraint_matrices((3, 3))
        with pytest.raises(ValueError):
            dysOpt(A, b, l, u, alpha=-0.1)
        with pytest.raises(ValueError):
            dysOpt(A, b, l, u, alpha=0)

    def test_construction_succeeds(self):
        from pyepo.func.unrolling import dysOpt
        A, b, l, u = _sp_constraint_matrices((3, 3))
        dys = dysOpt(A, b, l, u)
        assert dys.n == A.shape[1]
        assert dys.alpha == 0.05
        assert dys.max_iter == 1000


class TestDysOptProjections:

    def _make_dys(self, grid=(3, 3)):
        from pyepo.func.unrolling import dysOpt
        A, b, l, u = _sp_constraint_matrices(grid)
        return dysOpt(A, b, l, u, alpha=0.1, max_iter=500, tol=1e-4)

    def test_project_box(self):
        dys = self._make_dys()
        n = dys.n
        # build input with known out-of-bound values
        x = torch.zeros(1, n)
        x[0, 0] = -0.5  # below lb (0)
        x[0, 1] = 0.3   # within [0, 1]
        x[0, 2] = 1.5   # above ub (1)
        px = dys._project_box(x)
        # clamp to [0, 1]
        assert (px >= 0).all()
        assert (px <= 1).all()
        assert px[0, 0].item() == 0.0
        assert px[0, 1].item() == pytest.approx(0.3, abs=1e-6)
        assert px[0, 2].item() == 1.0

    def test_project_equality_satisfies_Ax_eq_b(self):
        dys = self._make_dys()
        z = torch.randn(4, dys.n)
        pz = dys._project_equality(z)
        # Ax should equal b for all instances
        residual = dys._A @ pz.T - dys._b.unsqueeze(1)
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-4)

    def test_project_equality_idempotent(self):
        dys = self._make_dys()
        z = torch.randn(2, dys.n)
        pz = dys._project_equality(z)
        ppz = dys._project_equality(pz)
        assert torch.allclose(pz, ppz, atol=1e-5)

    def test_project_equality_empty_A(self):
        """Empty equality constraints should return input unchanged."""
        from pyepo.func.unrolling import dysOpt
        n = 12
        A = np.zeros((0, n), dtype=np.float32)
        b = np.zeros(0, dtype=np.float32)
        l = np.zeros(n, dtype=np.float32)
        u = np.ones(n, dtype=np.float32)
        dys = dysOpt(A, b, l, u)
        z = torch.randn(3, n)
        pz = dys._project_equality(z)
        assert torch.allclose(pz, z)


class TestDysOptForward:

    def _make_dys(self, grid=(3, 3)):
        from pyepo.func.unrolling import dysOpt
        A, b, l, u = _sp_constraint_matrices(grid)
        return dysOpt(A, b, l, u, alpha=0.1, max_iter=1000, tol=1e-4)

    def test_output_shape(self):
        dys = self._make_dys()
        dys.eval()
        pred_cost = torch.rand(8, dys.n)
        with torch.no_grad():
            sol = dys(pred_cost)
        assert sol.shape == (8, dys.n)

    def test_output_within_bounds(self):
        dys = self._make_dys()
        dys.eval()
        pred_cost = torch.rand(4, dys.n)
        with torch.no_grad():
            sol = dys(pred_cost)
        assert (sol >= -1e-5).all()
        assert (sol <= 1 + 1e-5).all()

    def test_output_satisfies_constraints(self):
        dys = self._make_dys()
        dys.eval()
        pred_cost = torch.rand(4, dys.n)
        with torch.no_grad():
            sol = dys(pred_cost)
        residual = dys._A @ sol.T - dys._b.unsqueeze(1)
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-3)

    def test_gradient_flows(self):
        dys = self._make_dys()
        dys.train()
        pred_cost = torch.rand(4, dys.n, requires_grad=True)
        sol = dys(pred_cost)
        loss = (sol * pred_cost).sum()
        loss.backward()
        assert pred_cost.grad is not None
        assert (pred_cost.grad.abs() > 0).any()

    def test_gradient_nonzero_per_instance(self):
        dys = self._make_dys()
        dys.train()
        pred_cost = torch.rand(4, dys.n, requires_grad=True)
        sol = dys(pred_cost)
        loss = (sol * pred_cost).sum()
        loss.backward()
        # each instance should have at least one nonzero gradient
        for i in range(4):
            assert (pred_cost.grad[i].abs() > 1e-8).any()

    def test_batch_size_one(self):
        dys = self._make_dys()
        dys.train()
        pred_cost = torch.rand(1, dys.n, requires_grad=True)
        sol = dys(pred_cost)
        loss = sol.sum()
        loss.backward()
        assert pred_cost.grad is not None

    def test_maximize_sense(self):
        """Test that minimize=False produces valid solutions."""
        from pyepo.func.unrolling import dysOpt
        n = 4
        # simple equality: x_0 = 0.5
        A = np.zeros((1, n), dtype=np.float32)
        A[0, 0] = 1.0
        b = np.array([0.5], dtype=np.float32)
        l = np.zeros(n, dtype=np.float32)
        u = np.ones(n, dtype=np.float32)
        dys = dysOpt(A, b, l, u, alpha=0.1, max_iter=500, tol=1e-4, minimize=False)
        dys.train()
        pred_cost = torch.rand(2, n, requires_grad=True)
        sol = dys(pred_cost)
        assert sol.shape == (2, n)
        # x_0 should be close to 0.5
        assert torch.allclose(sol[:, 0], torch.tensor(0.5), atol=1e-2)
        # gradient should flow
        loss = (sol * pred_cost).sum()
        loss.backward()
        assert pred_cost.grad is not None
