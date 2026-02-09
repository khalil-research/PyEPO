#!/usr/bin/env python
# coding: utf-8
"""
Lightweight integration tests: end-to-end training for a few steps.

Uses tiny data (n=32) and minimal steps to verify the full pipeline
(data -> model -> loss -> backward -> metric) works without errors.
"""

import pytest
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import pyepo
from pyepo.data.dataset import optDataset, optDatasetKNN

try:
    from pyepo.model.grb.shortestpath import shortestPathModel
    from pyepo.model.grb.knapsack import knapsackModel
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

requires_gurobi = pytest.mark.skipif(not _HAS_GUROBI, reason="Gurobi not installed")


# ============================================================
# Helpers
# ============================================================

class _LinearModel(nn.Module):
    def __init__(self, num_feat, num_cost):
        super().__init__()
        self.linear = nn.Linear(num_feat, num_cost)

    def forward(self, x):
        return self.linear(x)


# small problem constants
_NUM_DATA = 32
_NUM_FEAT = 3
_GRID = (3, 3)  # 12 edges
_BATCH = 16
_STEPS = 3


def _train_loop(loss_fn, loader, predmodel, call):
    """Run a few training steps. `call` is a function(loss_fn, cp, c, w, z) -> scalar loss."""
    optimizer = torch.optim.SGD(predmodel.parameters(), lr=1e-2)
    for i, (x, c, w, z) in enumerate(loader):
        if i >= _STEPS:
            break
        cp = predmodel(x)
        loss = call(loss_fn, cp, c, w, z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ============================================================
# Shared fixture: shortest path data (reused by most tests)
# ============================================================

@pytest.fixture(scope="module")
def sp_data():
    if not _HAS_GUROBI:
        pytest.skip("Gurobi not installed")
    x, c = pyepo.data.shortestpath.genData(_NUM_DATA, _NUM_FEAT, _GRID, seed=42)
    optmodel = shortestPathModel(grid=_GRID)
    dataset = optDataset(optmodel, x, c)
    loader = DataLoader(dataset, batch_size=_BATCH, shuffle=False)
    return optmodel, dataset, loader


def _fresh_predmodel():
    return _LinearModel(_NUM_FEAT, 12)


# ============================================================
# Surrogate losses: SPOPlus, perturbationGradient
# ============================================================

@requires_gurobi
class TestSPOPlus:
    def test_train(self, sp_data):
        optmodel, dataset, loader = sp_data
        predmodel = _fresh_predmodel()
        spo = pyepo.func.SPOPlus(optmodel, processes=1)
        _train_loop(spo, loader, predmodel,
                    lambda fn, cp, c, w, z: fn(cp, c, w, z))

    def test_regret_metric(self, sp_data):
        optmodel, dataset, loader = sp_data
        predmodel = _fresh_predmodel()
        reg = pyepo.metric.regret(predmodel, optmodel, loader)
        assert isinstance(reg, float) and reg >= 0

    def test_mse_metric(self, sp_data):
        _, _, loader = sp_data
        predmodel = _fresh_predmodel()
        mse = pyepo.metric.MSE(predmodel, loader)
        assert isinstance(mse, float) and mse >= 0


@requires_gurobi
class TestPerturbationGradient:
    def test_train(self, sp_data):
        optmodel, dataset, loader = sp_data
        predmodel = _fresh_predmodel()
        pg = pyepo.func.perturbationGradient(optmodel, processes=1, sigma=1.0)
        _train_loop(pg, loader, predmodel,
                    lambda fn, cp, c, w, z: fn(cp, c))


# ============================================================
# Blackbox losses: blackboxOpt, negativeIdentity
# ============================================================

@requires_gurobi
class TestBlackboxOpt:
    def test_train(self, sp_data):
        optmodel, dataset, loader = sp_data
        predmodel = _fresh_predmodel()
        bb = pyepo.func.blackboxOpt(optmodel, processes=1, lambd=10)
        _train_loop(bb, loader, predmodel,
                    lambda fn, cp, c, w, z: fn(cp).mean())


@requires_gurobi
class TestNegativeIdentity:
    def test_train(self, sp_data):
        optmodel, dataset, loader = sp_data
        predmodel = _fresh_predmodel()
        nid = pyepo.func.negativeIdentity(optmodel, processes=1)
        _train_loop(nid, loader, predmodel,
                    lambda fn, cp, c, w, z: fn(cp).mean())


# ============================================================
# Perturbed losses: perturbedOpt, perturbedFenchelYoung,
#                   implicitMLE, adaptiveImplicitMLE
# ============================================================

@requires_gurobi
class TestPerturbedOpt:
    def test_train(self, sp_data):
        optmodel, dataset, loader = sp_data
        predmodel = _fresh_predmodel()
        ptb = pyepo.func.perturbedOpt(optmodel, processes=1, n_samples=3, sigma=1.0)
        _train_loop(ptb, loader, predmodel,
                    lambda fn, cp, c, w, z: fn(cp).mean())


@requires_gurobi
class TestPerturbedFenchelYoung:
    def test_train(self, sp_data):
        optmodel, dataset, loader = sp_data
        predmodel = _fresh_predmodel()
        pfy = pyepo.func.perturbedFenchelYoung(optmodel, processes=1, n_samples=3, sigma=1.0)
        _train_loop(pfy, loader, predmodel,
                    lambda fn, cp, c, w, z: fn(cp, w))


@requires_gurobi
class TestImplicitMLE:
    def test_train(self, sp_data):
        optmodel, dataset, loader = sp_data
        predmodel = _fresh_predmodel()
        imle = pyepo.func.implicitMLE(optmodel, processes=1, n_samples=3, sigma=1.0)
        _train_loop(imle, loader, predmodel,
                    lambda fn, cp, c, w, z: fn(cp).mean())


@requires_gurobi
class TestAdaptiveImplicitMLE:
    def test_train(self, sp_data):
        optmodel, dataset, loader = sp_data
        predmodel = _fresh_predmodel()
        aimle = pyepo.func.adaptiveImplicitMLE(optmodel, processes=1, n_samples=3, sigma=1.0)
        _train_loop(aimle, loader, predmodel,
                    lambda fn, cp, c, w, z: fn(cp).mean())


# ============================================================
# Contrastive losses: NCE, contrastiveMAP
# ============================================================

@requires_gurobi
class TestNCE:
    def test_train(self, sp_data):
        optmodel, dataset, loader = sp_data
        predmodel = _fresh_predmodel()
        nce = pyepo.func.NCE(optmodel, processes=1, solve_ratio=1, dataset=dataset)
        _train_loop(nce, loader, predmodel,
                    lambda fn, cp, c, w, z: fn(cp, w))


@requires_gurobi
class TestContrastiveMAP:
    def test_train(self, sp_data):
        optmodel, dataset, loader = sp_data
        predmodel = _fresh_predmodel()
        cmap = pyepo.func.contrastiveMAP(optmodel, processes=1, solve_ratio=1, dataset=dataset)
        _train_loop(cmap, loader, predmodel,
                    lambda fn, cp, c, w, z: fn(cp, w))


# ============================================================
# Ranking losses: listwiseLTR, pairwiseLTR, pointwiseLTR
# ============================================================

@requires_gurobi
class TestListwiseLTR:
    def test_train(self, sp_data):
        optmodel, dataset, loader = sp_data
        predmodel = _fresh_predmodel()
        lw = pyepo.func.listwiseLTR(optmodel, processes=1, solve_ratio=1, dataset=dataset)
        _train_loop(lw, loader, predmodel,
                    lambda fn, cp, c, w, z: fn(cp, c))


@requires_gurobi
class TestPairwiseLTR:
    def test_train(self, sp_data):
        optmodel, dataset, loader = sp_data
        predmodel = _fresh_predmodel()
        pw = pyepo.func.pairwiseLTR(optmodel, processes=1, solve_ratio=1, dataset=dataset)
        _train_loop(pw, loader, predmodel,
                    lambda fn, cp, c, w, z: fn(cp, c))


@requires_gurobi
class TestPointwiseLTR:
    def test_train(self, sp_data):
        optmodel, dataset, loader = sp_data
        predmodel = _fresh_predmodel()
        pt = pyepo.func.pointwiseLTR(optmodel, processes=1, solve_ratio=1, dataset=dataset)
        _train_loop(pt, loader, predmodel,
                    lambda fn, cp, c, w, z: fn(cp, c))


# ============================================================
# Knapsack (MAXIMIZE) integration
# ============================================================

@requires_gurobi
class TestKnapsackIntegration:

    def test_spo_plus(self):
        w_data, x, c = pyepo.data.knapsack.genData(
            _NUM_DATA, _NUM_FEAT, 4, dim=1, deg=1, seed=42)
        optmodel = knapsackModel(weights=w_data, capacity=[10.0])
        dataset = optDataset(optmodel, x, c)
        loader = DataLoader(dataset, batch_size=_BATCH, shuffle=False)
        predmodel = _LinearModel(_NUM_FEAT, optmodel.num_cost)

        spo = pyepo.func.SPOPlus(optmodel, processes=1)
        _train_loop(spo, loader, predmodel,
                    lambda fn, cp, c_b, w, z: fn(cp, c_b, w, z))

    def test_regret(self):
        w_data, x, c = pyepo.data.knapsack.genData(
            _NUM_DATA, _NUM_FEAT, 4, dim=1, deg=1, seed=42)
        optmodel = knapsackModel(weights=w_data, capacity=[10.0])
        dataset = optDataset(optmodel, x, c)
        loader = DataLoader(dataset, batch_size=_BATCH, shuffle=False)
        predmodel = _LinearModel(_NUM_FEAT, optmodel.num_cost)

        reg = pyepo.metric.regret(predmodel, optmodel, loader)
        assert isinstance(reg, float)


# ============================================================
# KNN dataset integration
# ============================================================

@requires_gurobi
class TestKNNIntegration:

    def test_knn_dataset_spo_plus(self):
        x, c = pyepo.data.shortestpath.genData(_NUM_DATA, _NUM_FEAT, _GRID, seed=42)
        optmodel = shortestPathModel(grid=_GRID)
        dataset = optDatasetKNN(optmodel, x, c, k=3, weight=0.5)
        loader = DataLoader(dataset, batch_size=_BATCH, shuffle=False)
        predmodel = _LinearModel(_NUM_FEAT, optmodel.num_cost)

        spo = pyepo.func.SPOPlus(optmodel, processes=1)
        _train_loop(spo, loader, predmodel,
                    lambda fn, cp, c_b, w, z: fn(cp, c_b, w, z))

    def test_knn_dataset_shapes(self):
        x, c = pyepo.data.shortestpath.genData(_NUM_DATA, _NUM_FEAT, _GRID, seed=42)
        optmodel = shortestPathModel(grid=_GRID)
        dataset = optDatasetKNN(optmodel, x, c, k=3, weight=0.5)
        assert len(dataset) == _NUM_DATA
        feat, cost, sol, obj = dataset[0]
        assert feat.shape == (_NUM_FEAT,)
        assert cost.shape == (optmodel.num_cost,)
        assert sol.shape == (optmodel.num_cost,)
        assert obj.shape == (1,)
