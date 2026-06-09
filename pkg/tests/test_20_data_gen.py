#!/usr/bin/env python
"""Tests for pyepo.data generators (knapsack, shortestpath, tsp, portfolio).

Pure numpy generation, no solver: shapes, dtypes, seed reproducibility,
``deg`` validation, and noise behaviour. Fast, deterministic, runs before any
solver layer.
"""

import numpy as np
import pytest

from pyepo.data import knapsack, portfolio, shortestpath, tsp


class TestKnapsackData:

    def test_output_shapes(self):
        weights, x, c = knapsack.genData(50, 5, 8, dim=2, deg=1, seed=42)
        assert weights.shape == (2, 8)
        assert x.shape == (50, 5)
        assert c.shape == (50, 8)

    def test_deterministic(self):
        w1, x1, c1 = knapsack.genData(20, 3, 4, seed=0)
        w2, x2, c2 = knapsack.genData(20, 3, 4, seed=0)
        np.testing.assert_array_equal(w1, w2)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(c1, c2)

    def test_different_seeds(self):
        _, _, c1 = knapsack.genData(20, 3, 4, seed=0)
        _, _, c2 = knapsack.genData(20, 3, 4, seed=1)
        assert not np.array_equal(c1, c2)

    def test_cost_dtype_float32(self):
        _, _, c = knapsack.genData(10, 3, 4)
        assert c.dtype == np.float32

    @pytest.mark.parametrize("deg", [0, -1, 1.5])
    def test_deg_invalid_raises(self, deg):
        with pytest.raises(ValueError):
            knapsack.genData(10, 3, 4, deg=deg)

    def test_higher_degree_finite(self):
        _, _, c = knapsack.genData(10, 3, 4, deg=3, seed=42)
        assert c.shape == (10, 4)
        assert np.all(np.isfinite(c))

    def test_noise_changes_costs(self):
        _, _, c0 = knapsack.genData(30, 3, 4, noise_width=0, seed=42)
        _, _, c1 = knapsack.genData(30, 3, 4, noise_width=0.5, seed=42)
        assert not np.array_equal(c0, c1)


class TestShortestPathData:

    def test_output_shapes(self):
        x, c = shortestpath.genData(50, 5, (4, 4), deg=1, seed=42)
        # edges = (4-1)*4 + (4-1)*4 = 24
        assert x.shape == (50, 5)
        assert c.shape == (50, 24)

    def test_edge_count_formula(self):
        grid = (3, 5)
        _x, c = shortestpath.genData(10, 3, grid, seed=42)
        assert c.shape[1] == (grid[0] - 1) * grid[1] + (grid[1] - 1) * grid[0]

    def test_deterministic(self):
        x1, c1 = shortestpath.genData(10, 3, (3, 3), seed=0)
        x2, c2 = shortestpath.genData(10, 3, (3, 3), seed=0)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(c1, c2)

    def test_different_seeds(self):
        _, c1 = shortestpath.genData(10, 3, (3, 3), seed=0)
        _, c2 = shortestpath.genData(10, 3, (3, 3), seed=1)
        assert not np.array_equal(c1, c2)

    def test_cost_dtype_float32(self):
        _, c = shortestpath.genData(10, 3, (3, 3))
        assert c.dtype == np.float32

    def test_positive_costs(self):
        _, c = shortestpath.genData(20, 5, (3, 3), seed=42)
        assert np.all(c > 0)

    def test_noise_changes_costs(self):
        _, c0 = shortestpath.genData(20, 5, (3, 3), noise_width=0, seed=42)
        _, c1 = shortestpath.genData(20, 5, (3, 3), noise_width=0.5, seed=42)
        assert not np.array_equal(c0, c1)

    @pytest.mark.parametrize("deg", [0, -1, 1.5])
    def test_deg_invalid_raises(self, deg):
        with pytest.raises(ValueError):
            shortestpath.genData(10, 3, (3, 3), deg=deg)


class TestTSPData:

    def test_output_shapes(self):
        x, c = tsp.genData(20, 5, 6, seed=42)
        # edges = 6*5/2 = 15
        assert x.shape == (20, 5)
        assert c.shape == (20, 15)

    def test_edge_count_formula(self):
        n = 8
        _x, c = tsp.genData(10, 3, n, seed=42)
        assert c.shape[1] == n * (n - 1) // 2

    def test_deterministic(self):
        x1, c1 = tsp.genData(10, 3, 5, seed=0)
        x2, c2 = tsp.genData(10, 3, 5, seed=0)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(c1, c2)

    def test_different_seeds(self):
        _, c1 = tsp.genData(10, 3, 5, seed=0)
        _, c2 = tsp.genData(10, 3, 5, seed=1)
        assert not np.array_equal(c1, c2)

    def test_cost_dtype_float32(self):
        _, c = tsp.genData(10, 3, 5)
        assert c.dtype == np.float32

    def test_noise_changes_costs(self):
        _, c0 = tsp.genData(20, 3, 5, noise_width=0, seed=42)
        _, c1 = tsp.genData(20, 3, 5, noise_width=0.5, seed=42)
        assert not np.array_equal(c0, c1)

    @pytest.mark.parametrize("deg", [0, -1, 1.5])
    def test_deg_invalid_raises(self, deg):
        with pytest.raises(ValueError):
            tsp.genData(10, 3, 5, deg=deg)


class TestPortfolioData:

    def test_output_shapes(self):
        cov, x, r = portfolio.genData(30, 5, 8, seed=42)
        assert cov.shape == (8, 8)
        assert x.shape == (30, 5)
        assert r.shape == (30, 8)

    def test_covariance_symmetric(self):
        cov, _, _ = portfolio.genData(10, 3, 6, seed=42)
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_covariance_psd(self):
        cov, _, _ = portfolio.genData(10, 3, 6, seed=42)
        assert np.all(np.linalg.eigvalsh(cov) >= -1e-10)

    def test_deterministic(self):
        cov1, x1, r1 = portfolio.genData(10, 3, 4, seed=0)
        cov2, x2, r2 = portfolio.genData(10, 3, 4, seed=0)
        np.testing.assert_array_equal(cov1, cov2)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(r1, r2)

    def test_deg_invalid_raises(self):
        with pytest.raises(ValueError):
            portfolio.genData(10, 3, 4, deg=0)
