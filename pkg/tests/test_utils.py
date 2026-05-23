#!/usr/bin/env python
# coding: utf-8
"""
Tests for pyepo.utils: getArgs utility, and pyepo.model.opt helpers.
"""

import pytest
import numpy as np
import torch

from pyepo.utils import getArgs
from pyepo.model.opt import costToNumpy, getTspTour, unionFind

try:
    from pyepo.model.grb.knapsack import knapsackModel
    from pyepo.model.grb.shortestpath import shortestPathModel
    _HAS_GUROBI = True
except (ImportError, NameError):
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


# ============================================================
# costToNumpy tests
# ============================================================

class TestCostToNumpy:

    def test_torch_tensor_detached(self):
        c = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        out = costToNumpy(c)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_torch_tensor_preserves_dtype(self):
        c = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        out = costToNumpy(c)
        assert out.dtype == np.float64

    def test_list_default_float32(self):
        out = costToNumpy([1, 2, 3])
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_numpy_array_dtype_cast(self):
        c = np.array([1, 2, 3], dtype=np.int64)
        out = costToNumpy(c, dtype=np.float64)
        assert out.dtype == np.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_tensor_moved_to_cpu(self):
        c = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        out = costToNumpy(c)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


# ============================================================
# getTspTour tests
# ============================================================

class TestGetTspTour:

    @staticmethod
    def _all_edges(n):
        # undirected edges in lexicographic order for a complete graph
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    def test_simple_4_node_tour(self):
        """0-1-2-3-0 active; other edges off."""
        edges = self._all_edges(4)  # 6 edges
        active = {(0, 1), (1, 2), (2, 3), (0, 3)}
        sol = [1.0 if e in active else 0.0 for e in edges]
        tour = getTspTour(edges, 4, sol)
        # tour must visit every node exactly once then close back to 0
        assert tour[0] == 0
        assert tour[-1] == 0
        assert sorted(tour[:-1]) == [0, 1, 2, 3]

    def test_threshold_filters_fractional(self):
        """Edges below threshold are ignored."""
        edges = self._all_edges(3)
        sol = [1.0, 1.0, 1.0]
        # threshold above 1.0 -> no edge active -> empty edges -> raises
        with pytest.raises(ValueError):
            getTspTour(edges, 3, sol, threshold=1.5)

    def test_missing_node_raises(self):
        """Solution missing a node entirely should raise."""
        edges = self._all_edges(4)
        # only one edge active, leaving nodes 2 and 3 uncovered
        sol = [1.0 if e == (0, 1) else 0.0 for e in edges]
        with pytest.raises(ValueError, match="cover all"):
            getTspTour(edges, 4, sol)

    def test_subtour_raises(self):
        """Disconnected subtours should raise."""
        edges = self._all_edges(4)
        # two disjoint 2-cycles: (0,1)(0,1) and (2,3)(2,3) -> infeasible
        active = {(0, 1), (2, 3)}
        sol = [1.0 if e in active else 0.0 for e in edges]
        with pytest.raises(ValueError):
            getTspTour(edges, 4, sol)


# ============================================================
# unionFind tests
# ============================================================

class TestUnionFind:

    def test_initial_each_singleton(self):
        uf = unionFind(5)
        for i in range(5):
            assert uf.find(i) == i

    def test_union_merges_components(self):
        uf = unionFind(5)
        uf.union(0, 1)
        uf.union(2, 3)
        assert uf.find(0) == uf.find(1)
        assert uf.find(2) == uf.find(3)
        assert uf.find(0) != uf.find(2)

    def test_union_returns_true_on_merge(self):
        uf = unionFind(3)
        assert uf.union(0, 1) is True
        # already connected -> no merge
        assert uf.union(0, 1) is False

    def test_path_compression(self):
        uf = unionFind(4)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)
        # all in one component
        root = uf.find(0)
        for i in range(4):
            assert uf.find(i) == root
