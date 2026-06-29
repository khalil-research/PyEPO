"""Expensive optimization-dataset fixtures shared across test modules."""

from __future__ import annotations

import pytest
from torch.utils.data import DataLoader

from tests.support.backends import (
    _HAS_GUROBI,
    requires_copt,
    requires_gurobi,
    requires_mpax,
    requires_ortools,
)
from tests.support.helpers import BATCH, GRID, NUM_DATA, NUM_FEAT


@pytest.fixture(scope="module")
def sp_data():
    """Shortest-path optDataset + loader (CPU)."""
    if not _HAS_GUROBI:
        pytest.skip("Gurobi not installed")
    import pyepo
    from pyepo.data.dataset import optDataset
    from pyepo.model.grb.shortestpath import shortestPathModel

    x, c = pyepo.data.shortestpath.genData(NUM_DATA, NUM_FEAT, GRID, seed=42)
    optmodel = shortestPathModel(grid=GRID)
    dataset = optDataset(optmodel, x, c)
    return optmodel, dataset, DataLoader(dataset, batch_size=BATCH, shuffle=False)


@pytest.fixture(scope="module")
def ks_data():
    """Knapsack optDataset + loader (CPU, MAXIMIZE)."""
    if not _HAS_GUROBI:
        pytest.skip("Gurobi not installed")
    import pyepo
    from pyepo.data.dataset import optDataset
    from pyepo.model.grb.knapsack import knapsackModel

    weights, x, c = pyepo.data.knapsack.genData(NUM_DATA, NUM_FEAT, 4, dim=1, deg=1, seed=42)
    optmodel = knapsackModel(weights=weights, capacity=[10.0])
    dataset = optDataset(optmodel, x, c)
    return optmodel, dataset, DataLoader(dataset, batch_size=BATCH, shuffle=False)


_PIPELINE_BACKENDS = [
    pytest.param("copt", marks=requires_copt),
    pytest.param("ort", marks=requires_ortools),
    pytest.param("mpax", marks=requires_mpax),
]
_SP_TRUTH_BACKENDS = [pytest.param("grb", marks=requires_gurobi), *_PIPELINE_BACKENDS]


def _sp_optmodel(backend):
    if backend == "grb":
        from pyepo.model.grb.shortestpath import shortestPathModel
    elif backend == "copt":
        from pyepo.model.copt.shortestpath import shortestPathModel
    elif backend == "ort":
        from pyepo.model.ort.shortestpath import shortestPathModel
    elif backend == "mpax":
        from pyepo.model.mpax.shortestpath import shortestPathModel
    else:
        raise ValueError(backend)
    return shortestPathModel(grid=GRID)


def _sp_dataset(backend):
    import pyepo
    from pyepo.data.dataset import optDataset

    optmodel = _sp_optmodel(backend)
    x, c = pyepo.data.shortestpath.genData(NUM_DATA, NUM_FEAT, GRID, seed=42)
    dataset = optDataset(optmodel, x, c)
    return optmodel, dataset, DataLoader(dataset, batch_size=BATCH, shuffle=False)


def _ks_optmodel(backend, weights):
    if backend == "grb":
        from pyepo.model.grb.knapsack import knapsackModel
    elif backend == "copt":
        from pyepo.model.copt.knapsack import knapsackModel
    elif backend == "ort":
        from pyepo.model.ort.knapsack import knapsackModel
    elif backend == "mpax":
        from pyepo.model.mpax.knapsack import knapsackModel
    else:
        raise ValueError(backend)
    return knapsackModel(weights=weights, capacity=[10.0])


@pytest.fixture(scope="module", params=_PIPELINE_BACKENDS)
def sp_pipeline(request):
    return _sp_dataset(request.param)


@pytest.fixture(scope="module", params=_SP_TRUTH_BACKENDS)
def sp_truth(request):
    return _sp_dataset(request.param)


@pytest.fixture(scope="module", params=_PIPELINE_BACKENDS)
def ks_pipeline(request):
    import pyepo
    from pyepo.data.dataset import optDataset

    weights, x, c = pyepo.data.knapsack.genData(NUM_DATA, NUM_FEAT, 4, dim=1, deg=1, seed=42)
    optmodel = _ks_optmodel(request.param, weights)
    dataset = optDataset(optmodel, x, c)
    return optmodel, dataset, DataLoader(dataset, batch_size=BATCH, shuffle=False)


@pytest.fixture(scope="module")
def sp_constrs_data():
    """Shortest-path binding-constraint dataset for CaVE."""
    if not _HAS_GUROBI:
        pytest.skip("Gurobi not installed")
    import pyepo
    from pyepo.data.dataset import optDataLoader, optDatasetConstrs
    from pyepo.model.grb.shortestpath import shortestPathModel

    x, c = pyepo.data.shortestpath.genData(8, NUM_FEAT, GRID, seed=42)
    optmodel = shortestPathModel(grid=GRID)
    dataset = optDatasetConstrs(optmodel, x, c)
    loader = optDataLoader(dataset, batch_size=4, shuffle=False)
    return optmodel, dataset, loader


__all__ = [
    "ks_data",
    "ks_pipeline",
    "sp_constrs_data",
    "sp_data",
    "sp_pipeline",
    "sp_truth",
]
