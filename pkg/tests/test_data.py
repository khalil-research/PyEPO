#!/usr/bin/env python
# coding: utf-8
"""
Tests for pyepo.data: data generation and dataset classes
"""

import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

from pyepo.data import knapsack, shortestpath, tsp, portfolio
from pyepo.data.dataset import (
    optDataset, optDatasetKNN, optDatasetConstrs, collate_tight_constraints,
)

try:
    from pyepo.model.grb.shortestpath import shortestPathModel
    from pyepo.model.grb.tsp import tspDFJModel
    # verify Gurobi actually works (import succeeds even without a valid license)
    shortestPathModel(grid=(3, 3))
    _HAS_GUROBI = True
except (ImportError, NameError, Exception):
    _HAS_GUROBI = False

requires_gurobi = pytest.mark.skipif(not _HAS_GUROBI, reason="Gurobi not installed")


# ============================================================
# Data generation tests
# ============================================================

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

    def test_cost_dtype(self):
        _, _, c = knapsack.genData(10, 3, 4)
        assert c.dtype == np.float32

    def test_deg_positive_int(self):
        with pytest.raises(ValueError):
            knapsack.genData(10, 3, 4, deg=0)
        with pytest.raises(ValueError):
            knapsack.genData(10, 3, 4, deg=-1)
        with pytest.raises(ValueError):
            knapsack.genData(10, 3, 4, deg=1.5)

    def test_higher_degree(self):
        _, _, c = knapsack.genData(10, 3, 4, deg=3, seed=42)
        assert c.shape == (10, 4)
        assert np.all(np.isfinite(c))

    def test_noise(self):
        _, _, c_no_noise = knapsack.genData(30, 3, 4, noise_width=0, seed=42)
        _, _, c_noise = knapsack.genData(30, 3, 4, noise_width=0.5, seed=42)
        # with noise_width=0, epsilon is uniform(1,1)=1, so no noise
        # values should differ when noise is added
        assert not np.array_equal(c_no_noise, c_noise)


class TestShortestPathData:

    def test_output_shapes(self):
        x, c = shortestpath.genData(50, 5, (4, 4), deg=1, seed=42)
        # edges = (4-1)*4 + (4-1)*4 = 24
        assert x.shape == (50, 5)
        assert c.shape == (50, 24)

    def test_edge_count(self):
        grid = (3, 5)
        x, c = shortestpath.genData(10, 3, grid, seed=42)
        expected_edges = (grid[0] - 1) * grid[1] + (grid[1] - 1) * grid[0]
        assert c.shape[1] == expected_edges

    def test_deterministic(self):
        x1, c1 = shortestpath.genData(10, 3, (3, 3), seed=0)
        x2, c2 = shortestpath.genData(10, 3, (3, 3), seed=0)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(c1, c2)

    def test_positive_costs(self):
        _, c = shortestpath.genData(20, 5, (3, 3), seed=42)
        assert np.all(c > 0)

    def test_deg_validation(self):
        with pytest.raises(ValueError):
            shortestpath.genData(10, 3, (3, 3), deg=0)


class TestTSPData:

    def test_output_shapes(self):
        x, c = tsp.genData(20, 5, 6, seed=42)
        # edges = 6*(6-1)/2 = 15
        assert x.shape == (20, 5)
        assert c.shape == (20, 15)

    def test_edge_count(self):
        n_nodes = 8
        x, c = tsp.genData(10, 3, n_nodes, seed=42)
        assert c.shape[1] == n_nodes * (n_nodes - 1) // 2

    def test_deterministic(self):
        x1, c1 = tsp.genData(10, 3, 5, seed=0)
        x2, c2 = tsp.genData(10, 3, 5, seed=0)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(c1, c2)

    def test_deg_validation(self):
        with pytest.raises(ValueError):
            tsp.genData(10, 3, 5, deg=-1)


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
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)

    def test_deterministic(self):
        cov1, x1, r1 = portfolio.genData(10, 3, 4, seed=0)
        cov2, x2, r2 = portfolio.genData(10, 3, 4, seed=0)
        np.testing.assert_array_equal(cov1, cov2)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(r1, r2)

    def test_deg_validation(self):
        with pytest.raises(ValueError):
            portfolio.genData(10, 3, 4, deg=0)


# ============================================================
# Dataset tests (mock solver to avoid Gurobi dependency)
# ============================================================

def _make_mock_optmodel(num_cost=4):
    """Create a mock optModel that returns dummy solutions."""
    model = MagicMock()
    model.num_cost = num_cost
    model.modelSense = 1  # MINIMIZE
    model.setObj = MagicMock()
    model.solve = MagicMock(return_value=(np.ones(num_cost), 1.0))
    # make isinstance check work
    type(model).__name__ = "MockOptModel"
    return model


class TestOptDataset:

    def test_create_dataset(self):
        num_cost = 4
        model = _make_mock_optmodel(num_cost)
        feats = np.random.randn(10, 3).astype(np.float32)
        costs = np.random.randn(10, num_cost).astype(np.float32)

        with patch("pyepo.data.dataset.optDataset.__init__", wraps=None) as mock_init:
            # bypass isinstance check by directly setting attributes
            ds = object.__new__(optDataset)
            ds.model = model
            ds.feats = feats
            ds.costs = costs
            # simulate _getSols
            sols = np.ones((10, num_cost), dtype=np.float32)
            objs = np.ones((10, 1), dtype=np.float32)
            ds.feats = torch.as_tensor(feats, dtype=torch.float32)
            ds.costs = torch.as_tensor(costs, dtype=torch.float32)
            ds.sols = torch.as_tensor(sols, dtype=torch.float32)
            ds.objs = torch.as_tensor(objs, dtype=torch.float32)

        assert len(ds) == 10

    def test_getitem_returns_tensors(self):
        ds = object.__new__(optDataset)
        n, d = 5, 3
        ds.feats = torch.randn(n, d)
        ds.costs = torch.randn(n, d)
        ds.sols = torch.randn(n, d)
        ds.objs = torch.randn(n, 1)

        x, c, w, z = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(c, torch.Tensor)
        assert isinstance(w, torch.Tensor)
        assert isinstance(z, torch.Tensor)

    def test_getitem_shapes(self):
        ds = object.__new__(optDataset)
        n, d = 8, 4
        ds.feats = torch.randn(n, d)
        ds.costs = torch.randn(n, d)
        ds.sols = torch.randn(n, d)
        ds.objs = torch.randn(n, 1)

        x, c, w, z = ds[2]
        assert x.shape == (d,)
        assert c.shape == (d,)
        assert w.shape == (d,)
        assert z.shape == (1,)

    def test_getitem_batch_indexing(self):
        ds = object.__new__(optDataset)
        n, d = 10, 3
        ds.feats = torch.randn(n, d)
        ds.costs = torch.randn(n, d)
        ds.sols = torch.randn(n, d)
        ds.objs = torch.randn(n, 1)

        x, c, w, z = ds[0:4]
        assert x.shape == (4, d)

    def test_len(self):
        ds = object.__new__(optDataset)
        ds.costs = torch.randn(15, 3)
        assert len(ds) == 15

    def test_tensors_are_float32(self):
        ds = object.__new__(optDataset)
        ds.feats = torch.as_tensor(np.random.randn(5, 3), dtype=torch.float32)
        ds.costs = torch.as_tensor(np.random.randn(5, 3), dtype=torch.float32)
        ds.sols = torch.as_tensor(np.random.randn(5, 3), dtype=torch.float32)
        ds.objs = torch.as_tensor(np.random.randn(5, 1), dtype=torch.float32)

        x, c, w, z = ds[0]
        assert x.dtype == torch.float32
        assert c.dtype == torch.float32
        assert w.dtype == torch.float32
        assert z.dtype == torch.float32


class TestOptDatasetInit:
    """Exercise the real __init__ path (not the object.__new__ bypass)."""

    def test_rejects_non_optmodel(self):
        x = np.random.randn(5, 3).astype(np.float32)
        c = np.random.randn(5, 4).astype(np.float32)
        with pytest.raises(TypeError):
            optDataset("not_a_model", x, c)


class TestOptDatasetKNN:

    def test_getKNN_vectorized(self):
        """Test that the vectorized _getKNN produces correct shapes."""
        ds = object.__new__(optDatasetKNN)
        n, d, k = 20, 5, 3
        ds.feats = np.random.randn(n, d).astype(np.float32)
        ds.costs = np.random.randn(n, d).astype(np.float32)
        ds.k = k
        ds.weight = 0.5

        costs_knn = ds._getKNN()
        assert costs_knn.shape == (n, d, k)

    def test_getKNN_weight_boundaries(self):
        """Test weight=1 means only self cost, weight=0 means only neighbors."""
        ds = object.__new__(optDatasetKNN)
        n, d, k = 10, 3, 2
        ds.feats = np.random.randn(n, d).astype(np.float32)
        ds.costs = np.random.randn(n, d).astype(np.float32)

        # weight=1: only self cost
        ds.k = k
        ds.weight = 1.0
        costs_knn = ds._getKNN()
        for i in range(n):
            for j in range(k):
                np.testing.assert_allclose(costs_knn[i, :, j], ds.costs[i])

    def test_getKNN_deterministic(self):
        ds = object.__new__(optDatasetKNN)
        n, d, k = 15, 4, 3
        ds.feats = np.random.RandomState(42).randn(n, d).astype(np.float32)
        ds.costs = np.random.RandomState(42).randn(n, d).astype(np.float32)
        ds.k = k
        ds.weight = 0.5

        result1 = ds._getKNN()
        result2 = ds._getKNN()
        np.testing.assert_array_equal(result1, result2)


# ============================================================
# optDatasetConstrs (CaVE binding-constraint dataset)
# ============================================================

@requires_gurobi
class TestOptDatasetConstrs:
    """Covers the binary-vertex guard and binding-constraint extraction."""

    def test_rejects_non_optmodel(self):
        x = np.random.randn(5, 3).astype(np.float32)
        c = np.random.randn(5, 12).astype(np.float32)
        with pytest.raises(TypeError):
            optDatasetConstrs("not_a_model", x, c)

    def test_rejects_non_binary_vertex(self):
        # shortestPath LP gives binary sols, but use a portfolio-style continuous
        # problem to trigger the binary-vertex check. Simpler: monkeypatch solve
        # to return a fractional vertex on the shortestPathModel.
        model = shortestPathModel(grid=(3, 3))
        num_cost = model.num_cost
        # override solve to return a fractional vertex
        def fake_solve():
            return np.full(num_cost, 0.5, dtype=np.float64), 0.0
        # patch the copy() returned model so each instance sees the fake
        original_copy = model.copy
        def patched_copy():
            m = original_copy()
            m.solve = fake_solve
            # bypass infeasibility check by stubbing optimize status
            m._model.optimize = lambda *a, **kw: None
            m._model.Status = 2  # GRB.OPTIMAL
            return m
        model.copy = patched_copy
        x = np.random.randn(2, 3).astype(np.float32)
        c = np.random.randn(2, num_cost).astype(np.float32)
        with pytest.raises(ValueError, match="binary"):
            optDatasetConstrs(model, x, c)

    def test_construct_with_shortestpath(self):
        # shortest-path is a BLP with binary vertices: should construct cleanly
        model = shortestPathModel(grid=(3, 3))
        x, c = shortestpath.genData(num_data=4, num_features=3, grid=(3, 3), seed=42)
        ds = optDatasetConstrs(model, x, c)
        assert len(ds) == 4
        # each instance carries a tight-constraint matrix
        assert len(ds.ctrs) == 4
        for ctrs_i in ds.ctrs:
            assert ctrs_i.shape[1] == model.num_cost

    def test_collate_pads_to_max_rows(self):
        # craft a tiny batch with ragged ctrs shapes
        x = torch.randn(2, 3)
        c = torch.randn(2, 4)
        w = torch.randn(2, 4)
        z = torch.randn(2, 1)
        ctrs0 = torch.randn(2, 4)
        ctrs1 = torch.randn(5, 4)
        batch = [
            (x[0], c[0], w[0], z[0], ctrs0),
            (x[1], c[1], w[1], z[1], ctrs1),
        ]
        feats, costs, sols, objs, padded = collate_tight_constraints(batch)
        # pad_sequence stacks to (B, max_rows, num_cost)
        assert padded.shape == (2, 5, 4)
        # short row is zero-padded at the tail
        assert torch.allclose(padded[0, 2:], torch.zeros(3, 4))
