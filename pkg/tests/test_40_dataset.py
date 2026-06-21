#!/usr/bin/env python
"""Tests for pyepo.data.dataset: optDataset, optDatasetKNN, optDatasetConstrs.

Exercises the real construction path (which solves one optimization per sample)
on tiny instances, the kNN smoothing and its k-range guard, the CaVE
binding-constraint dataset, and the error paths (bad model type, bad solve
return, non-binary vertex).
"""

import numpy as np
import pytest
import torch

from pyepo.data import shortestpath
from pyepo.data.dataset import (
    collate_tight_constraints,
    optDataset,
    optDatasetConstrs,
    optDatasetKNN,
)
from pyepo.model.opt import optModel

from .conftest import GRID, NUM_FEAT, requires_gurobi

_N = 6  # tiny: each sample triggers a solve at construction

_DATASET_BUILDERS = [
    pytest.param(optDataset, {}, id="standard"),
    pytest.param(optDatasetKNN, {"k": 1}, id="knn"),
    pytest.param(optDatasetConstrs, {}, id="constraints"),
]


# ============================================================
# A minimal concrete optModel for testing error paths without a solver
# ============================================================


class _GoodModel(optModel):
    """Returns a fixed feasible (sol, obj) pair."""

    def __init__(self, num_cost=4):
        self._num_cost = num_cost
        super().__init__()

    def _getModel(self):
        return None, list(range(self._num_cost))

    @property
    def num_cost(self):
        return self._num_cost

    def setObj(self, c):
        self._c = np.asarray(c, dtype=float)

    def solve(self):
        sol = np.ones(self._num_cost, dtype=np.float32)
        return sol, float(self._c @ sol)


class _BadReturnModel(_GoodModel):
    """solve() returns a single value instead of (sol, obj)."""

    def solve(self):
        return np.ones(self._num_cost, dtype=np.float32)


class _SolutionTypeModel(_GoodModel):
    """Returns the same solution through a selected public solve type."""

    def __init__(self, solution_type):
        self._solution_type = solution_type
        super().__init__()

    def solve(self):
        sol = self._solution_type(np.ones(self._num_cost, dtype=np.float32))
        return sol, float(self._c.sum())


# ============================================================
# optDataset
# ============================================================


class TestDatasetInputContract:
    @pytest.mark.parametrize(("dataset_cls", "kwargs"), _DATASET_BUILDERS)
    def test_rejects_non_optmodel(self, dataset_cls, kwargs):
        rng = np.random.RandomState(0)
        x = rng.randn(_N, NUM_FEAT).astype(np.float32)
        c = rng.randn(_N, 4).astype(np.float32)
        with pytest.raises(TypeError):
            dataset_cls("not_a_model", x, c, **kwargs)

    @pytest.mark.parametrize(("dataset_cls", "kwargs"), _DATASET_BUILDERS)
    def test_rejects_length_mismatch(self, dataset_cls, kwargs):
        rng = np.random.RandomState(0)
        x = rng.randn(3, NUM_FEAT).astype(np.float32)
        c = rng.randn(4, 4).astype(np.float32)
        with pytest.raises(ValueError, match="same number of instances"):
            dataset_cls(_GoodModel(4), x, c, **kwargs)


class TestOptDatasetErrors:
    def test_bad_solve_return_raises(self):
        rng = np.random.RandomState(0)
        x = rng.randn(_N, NUM_FEAT).astype(np.float32)
        c = rng.randn(_N, 4).astype(np.float32)
        # a malformed solve() return surfaces as the raw unpacking error
        with pytest.raises((TypeError, ValueError)):
            optDataset(_BadReturnModel(4), x, c)


class TestOptDatasetConcrete:
    """Real construction path using the minimal concrete model (no solver)."""

    @pytest.fixture
    def ds(self):
        x = np.random.RandomState(0).randn(_N, NUM_FEAT).astype(np.float32)
        c = np.random.RandomState(1).randn(_N, 4).astype(np.float32)
        return optDataset(_GoodModel(4), x, c)

    def test_len(self, ds):
        assert len(ds) == _N

    def test_solved_shapes(self, ds):
        assert ds.sols.shape == (_N, 4)
        assert ds.objs.shape == (_N, 1)

    def test_getitem_four_tensors(self, ds):
        x, c, w, z = ds[2]
        for t in (x, c, w, z):
            assert isinstance(t, torch.Tensor)
            assert t.dtype == torch.float32
        assert x.shape == (NUM_FEAT,)
        assert c.shape == (4,)
        assert w.shape == (4,)
        assert z.shape == (1,)

    def test_objective_consistent_with_solution(self, ds):
        # _GoodModel returns sol=ones, obj=c·ones
        np.testing.assert_allclose(
            ds.objs.numpy().ravel(),
            ds.costs.numpy().sum(axis=1),
            atol=1e-4,
        )

    @pytest.mark.parametrize("solution_type", [np.asarray, torch.as_tensor, list])
    def test_normalizes_supported_solution_types(self, solution_type):
        x = np.zeros((2, NUM_FEAT), dtype=np.float32)
        c = np.ones((2, 4), dtype=np.float32)
        ds = optDataset(_SolutionTypeModel(solution_type), x, c)

        assert ds.sols.dtype == torch.float32
        torch.testing.assert_close(ds.sols, torch.ones((2, 4)))


@requires_gurobi
class TestOptDatasetGurobi:
    """Real solving via Gurobi: optimal objective equals cost·solution."""

    def test_solution_optimal(self):
        from pyepo.model.grb.shortestpath import shortestPathModel

        x, c = shortestpath.genData(_N, NUM_FEAT, GRID, seed=42)
        model = shortestPathModel(grid=GRID)
        ds = optDataset(model, x, c)
        # objs must equal costs · sols
        recon = (ds.costs * ds.sols).sum(dim=1)
        np.testing.assert_allclose(recon.numpy(), ds.objs.numpy().ravel(), atol=1e-4)


# ============================================================
# optDatasetKNN
# ============================================================


@requires_gurobi
class TestOptDatasetKNN:
    def _model_data(self, n=12):
        from pyepo.model.grb.shortestpath import shortestPathModel

        x, c = shortestpath.genData(n, NUM_FEAT, GRID, seed=42)
        return shortestPathModel(grid=GRID), x, c

    @pytest.mark.parametrize("k", [0, -1])
    def test_k_below_one_raises(self, k):
        model, x, c = self._model_data()
        with pytest.raises(ValueError, match="positive integer"):
            optDatasetKNN(model, x, c, k=k)

    @pytest.mark.parametrize("k", [1.5, True])
    def test_k_must_be_an_integer(self, k):
        model, x, c = self._model_data()
        with pytest.raises(ValueError, match="positive integer"):
            optDatasetKNN(model, x, c, k=k)

    def test_k_equal_num_data_raises(self):
        # guard is 1 <= k < num_data
        model, x, c = self._model_data(n=12)
        with pytest.raises(ValueError, match="1 <= k"):
            optDatasetKNN(model, x, c, k=12)

    @pytest.mark.parametrize("weight", [-0.1, 1.1, np.nan, np.inf, True])
    def test_invalid_weight_raises(self, weight):
        model, x, c = self._model_data()
        with pytest.raises(ValueError, match="weight"):
            optDatasetKNN(model, x, c, k=3, weight=weight)

    def test_valid_k_constructs(self):
        model, x, c = self._model_data(n=12)
        ds = optDatasetKNN(model, x, c, k=3, weight=0.5)
        assert len(ds) == 12
        _feat, cost, sol, obj = ds[0]
        assert cost.shape == (model.num_cost,)
        assert sol.shape == (model.num_cost,)
        assert obj.shape == (1,)

    def test_getKNN_shape_and_weight_boundary(self):
        # weight=1 => smoothed cost equals the self cost for every neighbour
        ds = object.__new__(optDatasetKNN)
        n, d, k = 10, 3, 2
        ds.feats = np.random.RandomState(0).randn(n, d).astype(np.float32)
        ds.costs = np.random.RandomState(1).randn(n, d).astype(np.float32)
        ds.k, ds.weight = k, 1.0
        knn = ds._getKNN()
        assert knn.shape == (n, d, k)
        for i in range(n):
            for j in range(k):
                np.testing.assert_allclose(knn[i, :, j], ds.costs[i])


# ============================================================
# optDatasetConstrs (CaVE binding-constraint dataset)
# ============================================================


@requires_gurobi
class TestOptDatasetConstrs:
    def test_construct_shortestpath(self):
        from pyepo.model.grb.shortestpath import shortestPathModel

        model = shortestPathModel(grid=GRID)
        x, c = shortestpath.genData(4, NUM_FEAT, GRID, seed=42)
        ds = optDatasetConstrs(model, x, c)
        assert len(ds) == 4
        assert len(ds.ctrs) == 4
        for ctrs_i in ds.ctrs:
            assert ctrs_i.shape[1] == model.num_cost

    def test_rejects_non_binary_vertex(self):
        from pyepo.model.grb.shortestpath import shortestPathModel

        model = shortestPathModel(grid=GRID)
        num_cost = model.num_cost

        def fake_solve():
            return np.full(num_cost, 0.5, dtype=np.float64), 0.0

        # gurobipy.Model.__setattr__ blocks method override; proxy fakes Status
        class _StatusProxy:
            def __init__(self, real, status):
                object.__setattr__(self, "_real", real)
                object.__setattr__(self, "_fake_status", status)

            @property
            def Status(self):
                return self._fake_status

            def optimize(self, *a, **kw):
                pass

            def __getattr__(self, name):
                return getattr(self._real, name)

            def __setattr__(self, name, value):
                setattr(self._real, name, value)

        model.solve = fake_solve
        model._model = _StatusProxy(model._model, 2)  # 2 == GRB.OPTIMAL
        rng = np.random.RandomState(0)
        x = rng.randn(2, NUM_FEAT).astype(np.float32)
        c = rng.randn(2, num_cost).astype(np.float32)
        with pytest.raises(ValueError, match="binary"):
            optDatasetConstrs(model, x, c)


class TestCollateTightConstraints:
    """Pure: ragged binding-constraint matrices pad to a common batch shape."""

    def test_pads_to_max_rows(self):
        x, c = torch.randn(2, 3), torch.randn(2, 4)
        w, z = torch.randn(2, 4), torch.randn(2, 1)
        ctrs0, ctrs1 = torch.randn(2, 4), torch.randn(5, 4)
        batch = [
            (x[0], c[0], w[0], z[0], ctrs0),
            (x[1], c[1], w[1], z[1], ctrs1),
        ]
        _f, _c, _s, _o, padded = collate_tight_constraints(batch)
        assert padded.shape == (2, 5, 4)
        # the shorter matrix is zero-padded at the tail
        assert torch.allclose(padded[0, 2:], torch.zeros(3, 4))
