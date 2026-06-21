#!/usr/bin/env python
"""Tests for pyepo.utils and pyepo.model.utils helpers.

Pure-logic layer: unionFind, getTspTour, costToNumpy need no solver and run
fast. getArgs needs a real optModel (Gurobi) and is gated accordingly.
"""

import pickle

import numpy as np
import pytest
import torch

from pyepo.model.opt import ModelSpec, optModel
from pyepo.model.utils import getTspTour, unionFind
from pyepo.utils import costToNumpy, getArgs

from .conftest import requires_gurobi


class ConfigModel(optModel):
    """Solver-free model for the explicit reconstruction protocol."""

    def __init__(self, values, label="default"):
        self.values = np.asarray(values)
        self.label = label
        super().__init__()

    def get_config(self):
        return {
            **super().get_config(),
            "values": self.values,
            "label": self.label,
        }

    def _getModel(self):
        return None, list(range(len(self.values)))

    def setObj(self, c):
        self.cost = np.asarray(c)

    def solve(self):
        return np.zeros(len(self.values)), 0.0


# ============================================================
# unionFind (pure)
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
        root = uf.find(0)
        for i in range(4):
            assert uf.find(i) == root


# ============================================================
# getTspTour (pure)
# ============================================================


class TestGetTspTour:
    @staticmethod
    def _all_edges(n):
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    def test_simple_4_node_tour(self):
        edges = self._all_edges(4)  # 6 edges
        active = {(0, 1), (1, 2), (2, 3), (0, 3)}
        sol = [1.0 if e in active else 0.0 for e in edges]
        tour = getTspTour(edges, 4, sol)
        assert tour[0] == 0
        assert tour[-1] == 0
        assert sorted(tour[:-1]) == [0, 1, 2, 3]

    def test_accepts_tensor_solution(self):
        edges = self._all_edges(4)
        active = {(0, 1), (1, 2), (2, 3), (0, 3)}
        sol = torch.tensor([1.0 if e in active else 0.0 for e in edges])
        tour = getTspTour(edges, 4, sol)
        assert sorted(tour[:-1]) == [0, 1, 2, 3]

    def test_threshold_filters_fractional(self):
        edges = self._all_edges(3)
        sol = [1.0, 1.0, 1.0]
        # threshold above 1.0 -> no active edge -> raises
        with pytest.raises(ValueError):
            getTspTour(edges, 3, sol, threshold=1.5)

    def test_missing_node_raises(self):
        edges = self._all_edges(4)
        sol = [1.0 if e == (0, 1) else 0.0 for e in edges]
        with pytest.raises(ValueError, match="cover all"):
            getTspTour(edges, 4, sol)

    def test_subtour_raises(self):
        edges = self._all_edges(4)
        # two disjoint edges, no Hamiltonian cycle
        active = {(0, 1), (2, 3)}
        sol = [1.0 if e in active else 0.0 for e in edges]
        with pytest.raises(ValueError):
            getTspTour(edges, 4, sol)


# ============================================================
# costToNumpy (pure)
# ============================================================


class TestCostToNumpy:
    def test_torch_tensor_detached(self):
        c = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        out = costToNumpy(c)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_torch_tensor_preserves_dtype(self):
        c = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        assert costToNumpy(c).dtype == np.float64

    def test_list_default_float32(self):
        out = costToNumpy([1, 2, 3])
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_numpy_array_dtype_cast(self):
        c = np.array([1, 2, 3], dtype=np.int64)
        assert costToNumpy(c, dtype=np.float64).dtype == np.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_tensor_moved_to_cpu(self):
        c = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        out = costToNumpy(c)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


# ============================================================
# explicit model reconstruction protocol (pure)
# ============================================================


class TestModelSpec:
    def test_get_config_and_compatibility_wrapper(self):
        model = ConfigModel([1, 2, 3], label="x")
        config = model.get_config()
        assert getArgs(model) == config
        assert config["label"] == "x"
        np.testing.assert_array_equal(config["values"], [1, 2, 3])

    def test_rebuild_has_clean_independent_config(self):
        model = ConfigModel([1, 2, 3], label="x")
        rebuilt = model.rebuild()

        assert type(rebuilt) is type(model)
        assert rebuilt is not model
        assert rebuilt.values is not model.values
        np.testing.assert_array_equal(rebuilt.values, model.values)
        assert rebuilt.label == model.label
        assert not hasattr(rebuilt, "cost")

    def test_spec_snapshots_mutable_config(self):
        model = ConfigModel([1, 2, 3])
        spec = model.to_spec()
        model.values[0] = 99

        rebuilt = spec.build()
        np.testing.assert_array_equal(rebuilt.values, [1, 2, 3])

    def test_spec_config_export_is_independent(self):
        spec = ConfigModel([1, 2, 3]).to_spec()
        exported = spec.config
        exported["values"][0] = 99

        np.testing.assert_array_equal(spec.config["values"], [1, 2, 3])
        np.testing.assert_array_equal(spec.build().values, [1, 2, 3])

    def test_spec_is_pickleable(self):
        model = ConfigModel([1, 2, 3], label="pickled")
        spec = pickle.loads(pickle.dumps(model.to_spec()))

        assert isinstance(spec, ModelSpec)
        rebuilt = spec.build()
        assert rebuilt.label == "pickled"
        np.testing.assert_array_equal(rebuilt.values, model.values)

    def test_worker_initializer_builds_from_spec(self):
        from pyepo.func.utils import (
            _init_worker_model,
            _solve_with_obj_in_worker,
        )

        model = ConfigModel([1, 2, 3])
        _init_worker_model(model.to_spec())
        sol, obj = _solve_with_obj_in_worker(np.array([3.0, 2.0, 1.0]))

        np.testing.assert_array_equal(sol, [0.0, 0.0, 0.0])
        assert obj == 0.0


# ============================================================
# getArgs (needs a real optModel)
# ============================================================


@requires_gurobi
class TestGetArgs:
    def _models(self):
        from pyepo.model.grb.knapsack import knapsackModel
        from pyepo.model.grb.shortestpath import shortestPathModel

        return knapsackModel, shortestPathModel

    def test_knapsack_args(self):
        knapsackModel, _ = self._models()
        weights = np.array([[3.0, 4.0, 5.0]])
        capacity = np.array([10.0])
        args = getArgs(knapsackModel(weights=weights, capacity=capacity))
        assert "weights" in args and "capacity" in args
        np.testing.assert_array_equal(args["weights"], weights)
        np.testing.assert_array_equal(args["capacity"], capacity)

    def test_shortestpath_args(self):
        _, shortestPathModel = self._models()
        args = getArgs(shortestPathModel(grid=(4, 4)))
        assert args["grid"] == (4, 4)

    def test_no_internal_state(self):
        _, shortestPathModel = self._models()
        args = getArgs(shortestPathModel(grid=(3, 3)))
        # introspects __init__ params only, not derived attrs / solver handle
        assert "arcs" not in args
        assert "_model" not in args

    def test_rebuild_solves_same(self):
        _, shortestPathModel = self._models()
        model = shortestPathModel(grid=(3, 3))
        model2 = model.rebuild()
        cost = np.random.RandomState(42).rand(model.num_cost)
        model.setObj(cost)
        sol1, obj1 = model.solve()
        model2.setObj(cost)
        sol2, obj2 = model2.solve()
        np.testing.assert_allclose(obj1, obj2, atol=1e-6)
        np.testing.assert_allclose(sol1, sol2, atol=1e-6)
