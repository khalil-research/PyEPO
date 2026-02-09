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
from pyepo.func.utils import _cache_in_pass, _check_sol, sumGammaDistribution

try:
    from pyepo.model.grb.shortestpath import shortestPathModel
    from pyepo.model.grb.knapsack import knapsackModel
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

requires_gurobi = pytest.mark.skipif(not _HAS_GUROBI, reason="Gurobi not installed")


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
        sol, obj = _cache_in_pass(cp, model, solpool)
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
        sol, obj = _cache_in_pass(cp, model, solpool)
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
        sol, obj = _cache_in_pass(cp, model, solpool)
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

    def _make_module(self):
        """Create a bare optModule-like object for pool testing."""
        from pyepo.func.abcmodule import optModule
        # We can't instantiate optModule (abstract), so make a minimal mock
        obj = object.__new__(optModule)
        obj.solpool = None
        obj._sol_set = set()
        return obj

    def test_init_pool(self):
        mod = self._make_module()
        sol = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        mod._update_solution_pool(sol)
        assert mod.solpool.shape == (2, 2)
        assert len(mod._sol_set) == 2

    def test_dedup(self):
        mod = self._make_module()
        sol1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        mod._update_solution_pool(sol1)
        # add same solutions again
        sol2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        mod._update_solution_pool(sol2)
        # pool should still have only 2 solutions
        assert mod.solpool.shape[0] == 2
        assert len(mod._sol_set) == 2

    def test_add_new(self):
        mod = self._make_module()
        sol1 = torch.tensor([[1.0, 0.0]])
        mod._update_solution_pool(sol1)
        sol2 = torch.tensor([[0.0, 1.0]])
        mod._update_solution_pool(sol2)
        assert mod.solpool.shape[0] == 2

    def test_mixed_new_and_dup(self):
        mod = self._make_module()
        sol1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        mod._update_solution_pool(sol1)
        # one new, one duplicate
        sol2 = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        mod._update_solution_pool(sol2)
        assert mod.solpool.shape[0] == 3
        assert len(mod._sol_set) == 3

    def test_empty_new_after_dedup(self):
        mod = self._make_module()
        sol = torch.tensor([[1.0, 2.0]])
        mod._update_solution_pool(sol)
        # all duplicates
        mod._update_solution_pool(sol)
        mod._update_solution_pool(sol)
        assert mod.solpool.shape[0] == 1


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
