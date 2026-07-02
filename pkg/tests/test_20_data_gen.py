#!/usr/bin/env python
"""Tests for pyepo.data generators (knapsack, shortestpath, tsp, portfolio).

Pure numpy generation, no solver: shapes, dtypes, seed reproducibility,
``deg`` validation, and noise behaviour. Fast, deterministic, runs before any
solver layer. Behaviours shared by every generator are parametrized over
``_GENERATORS``; the per-generator classes keep only generator-specific facts.
"""

import numpy as np
import pytest

from pyepo.data import knapsack, portfolio, shortestpath, tsp

_GENERATORS = [
    pytest.param(knapsack.genData, (10, 3, 4), id="knapsack"),
    pytest.param(shortestpath.genData, (10, 3, (3, 3)), id="shortestpath"),
    pytest.param(tsp.genData, (10, 3, 5), id="tsp"),
    pytest.param(portfolio.genData, (10, 3, 4), id="portfolio"),
]

# generator plus the name of its noise kwarg
_NOISE_GENERATORS = [
    pytest.param(knapsack.genData, (10, 3, 4), "noise_width", id="knapsack"),
    pytest.param(shortestpath.genData, (10, 3, (3, 3)), "noise_width", id="shortestpath"),
    pytest.param(tsp.genData, (10, 3, 5), "noise_width", id="tsp"),
    pytest.param(portfolio.genData, (10, 3, 4), "noise_level", id="portfolio"),
]


@pytest.mark.parametrize(("generator", "args"), _GENERATORS)
@pytest.mark.parametrize("deg", [0, -1, 1.5, True])
def test_invalid_degree_rejected(generator, args, deg):
    with pytest.raises(ValueError):
        generator(*args, deg=deg)


@pytest.mark.parametrize(("generator", "args", "noise_kw"), _NOISE_GENERATORS)
@pytest.mark.parametrize("noise", [-0.1, np.nan, np.inf, True])
def test_invalid_noise_rejected(generator, args, noise_kw, noise):
    with pytest.raises(ValueError, match=noise_kw):
        generator(*args, **{noise_kw: noise})


@pytest.mark.parametrize(("generator", "args"), _GENERATORS)
def test_seed_reproducibility(generator, args):
    # same seed reproduces every output array
    out1 = generator(*args, seed=0)
    out2 = generator(*args, seed=0)
    for a1, a2 in zip(out1, out2):
        np.testing.assert_array_equal(a1, a2)
    # a different seed changes the costs (last output)
    out3 = generator(*args, seed=1)
    assert not np.array_equal(out1[-1], out3[-1])


@pytest.mark.parametrize(("generator", "args"), _GENERATORS)
def test_cost_dtype_float32(generator, args):
    assert generator(*args)[-1].dtype == np.float32


@pytest.mark.parametrize(("generator", "args"), _GENERATORS)
def test_higher_degree_finite(generator, args):
    assert np.all(np.isfinite(generator(*args, deg=3, seed=42)[-1]))


@pytest.mark.parametrize(("generator", "args", "noise_kw"), _NOISE_GENERATORS)
def test_noise_changes_costs(generator, args, noise_kw):
    c0 = generator(*args, **{noise_kw: 0}, seed=42)[-1]
    c1 = generator(*args, **{noise_kw: 0.5}, seed=42)[-1]
    assert not np.array_equal(c0, c1)


class TestKnapsackData:
    """Knapsack generator shapes."""

    def test_output_shapes(self):
        weights, x, c = knapsack.genData(50, 5, 8, dim=2, deg=1, seed=42)
        assert weights.shape == (2, 8)
        assert x.shape == (50, 5)
        assert c.shape == (50, 8)


class TestShortestPathData:
    """Shortest-path generator shapes and positivity."""

    @pytest.mark.parametrize("grid", [(4, 4), (3, 5)])
    def test_output_shapes(self, grid):
        x, c = shortestpath.genData(20, 5, grid, deg=1, seed=42)
        assert x.shape == (20, 5)
        # directed grid arcs: down + right
        assert c.shape[1] == (grid[0] - 1) * grid[1] + (grid[1] - 1) * grid[0]

    def test_positive_costs(self):
        _, c = shortestpath.genData(20, 5, (3, 3), seed=42)
        assert np.all(c > 0)


class TestTSPData:
    """TSP generator shapes."""

    @pytest.mark.parametrize("num_nodes", [6, 8])
    def test_output_shapes(self, num_nodes):
        x, c = tsp.genData(20, 5, num_nodes, seed=42)
        assert x.shape == (20, 5)
        # one cost per undirected edge
        assert c.shape[1] == num_nodes * (num_nodes - 1) // 2


class TestPortfolioData:
    """Portfolio generator shapes and covariance."""

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
