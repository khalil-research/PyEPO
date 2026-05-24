#!/usr/bin/env python
# coding: utf-8
"""
Lightweight integration tests: end-to-end training for a few steps.

Uses tiny data (n=32) and minimal steps to verify the full pipeline
(data -> model -> loss -> backward -> metric) works without errors.

Shared fixtures (sp_data, LinearPred, constants) live in conftest.py.
"""

import pytest
import torch
from torch.utils.data import DataLoader

import pyepo
from pyepo.data.dataset import optDatasetKNN

from .conftest import (
    LinearPred, NUM_DATA, NUM_FEAT, GRID, BATCH, _HAS_GUROBI,
)

try:
    from pyepo.model.grb.shortestpath import shortestPathModel
except (ImportError, NameError):
    pass

requires_gurobi = pytest.mark.skipif(not _HAS_GUROBI, reason="Gurobi not installed")

# how many training steps to take per integration test
_STEPS = 3


# ============================================================
# Helpers
# ============================================================

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


def _fresh_predmodel(num_cost=12):
    return LinearPred(NUM_FEAT, num_cost)


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
# Perturbed losses: perturbedOpt, perturbedOptMul,
#                   perturbedFenchelYoung, perturbedFenchelYoungMul,
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
class TestPerturbedOptMul:
    def test_train(self, sp_data):
        optmodel, dataset, loader = sp_data
        predmodel = _fresh_predmodel()
        ptb = pyepo.func.perturbedOptMul(optmodel, processes=1, n_samples=3, sigma=0.5)
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
class TestPerturbedFenchelYoungMul:
    def test_train(self, sp_data):
        optmodel, dataset, loader = sp_data
        predmodel = _fresh_predmodel()
        pfy = pyepo.func.perturbedFenchelYoungMul(optmodel, processes=1, n_samples=3, sigma=0.5)
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
# Parallel solving (processes > 1)
# ============================================================

@requires_gurobi
class TestParallelSolving:

    def test_spo_plus_parallel(self, sp_data):
        optmodel, dataset, loader = sp_data
        predmodel = _fresh_predmodel()
        spo = pyepo.func.SPOPlus(optmodel, processes=2)
        _train_loop(spo, loader, predmodel,
                    lambda fn, cp, c, w, z: fn(cp, c, w, z))

    def test_blackbox_parallel(self, sp_data):
        optmodel, dataset, loader = sp_data
        predmodel = _fresh_predmodel()
        bb = pyepo.func.blackboxOpt(optmodel, processes=2, lambd=10)
        _train_loop(bb, loader, predmodel,
                    lambda fn, cp, c, w, z: fn(cp).mean())

    def test_perturbed_opt_parallel(self, sp_data):
        optmodel, dataset, loader = sp_data
        predmodel = _fresh_predmodel()
        ptb = pyepo.func.perturbedOpt(optmodel, processes=2, n_samples=3, sigma=1.0)
        _train_loop(ptb, loader, predmodel,
                    lambda fn, cp, c, w, z: fn(cp).mean())


# ============================================================
# Knapsack (MAXIMIZE) integration
# ============================================================

@requires_gurobi
class TestKnapsackIntegration:

    def test_spo_plus(self, ks_data):
        optmodel, dataset, loader = ks_data
        predmodel = LinearPred(NUM_FEAT, optmodel.num_cost)
        spo = pyepo.func.SPOPlus(optmodel, processes=1)
        _train_loop(spo, loader, predmodel,
                    lambda fn, cp, c_b, w, z: fn(cp, c_b, w, z))

    def test_regret(self, ks_data):
        optmodel, dataset, loader = ks_data
        predmodel = LinearPred(NUM_FEAT, optmodel.num_cost)
        reg = pyepo.metric.regret(predmodel, optmodel, loader)
        assert isinstance(reg, float)


# ============================================================
# KNN dataset integration
# ============================================================

@requires_gurobi
class TestKNNIntegration:

    def test_knn_dataset_spo_plus(self):
        x, c = pyepo.data.shortestpath.genData(NUM_DATA, NUM_FEAT, GRID, seed=42)
        optmodel = shortestPathModel(grid=GRID)
        dataset = optDatasetKNN(optmodel, x, c, k=3, weight=0.5)
        loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)
        predmodel = LinearPred(NUM_FEAT, optmodel.num_cost)

        spo = pyepo.func.SPOPlus(optmodel, processes=1)
        _train_loop(spo, loader, predmodel,
                    lambda fn, cp, c_b, w, z: fn(cp, c_b, w, z))

    def test_knn_dataset_shapes(self):
        x, c = pyepo.data.shortestpath.genData(NUM_DATA, NUM_FEAT, GRID, seed=42)
        optmodel = shortestPathModel(grid=GRID)
        dataset = optDatasetKNN(optmodel, x, c, k=3, weight=0.5)
        assert len(dataset) == NUM_DATA
        feat, cost, sol, obj = dataset[0]
        assert feat.shape == (NUM_FEAT,)
        assert cost.shape == (optmodel.num_cost,)
        assert sol.shape == (optmodel.num_cost,)
        assert obj.shape == (1,)


# ============================================================
# Portfolio QP integration (exercises QP setObj path)
# ============================================================

@requires_gurobi
class TestPortfolioIntegration:

    def _setup(self, num_assets=8):
        from pyepo.data.dataset import optDataset
        from pyepo.model.grb.portfolio import portfolioModel
        cov, x, c = pyepo.data.portfolio.genData(
            NUM_DATA, NUM_FEAT, num_assets, deg=1, seed=42)
        optmodel = portfolioModel(num_assets=num_assets, covariance=cov)
        dataset = optDataset(optmodel, x, c)
        loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)
        return optmodel, loader

    def test_spo_plus(self):
        optmodel, loader = self._setup()
        predmodel = LinearPred(NUM_FEAT, optmodel.num_cost)
        spo = pyepo.func.SPOPlus(optmodel, processes=1)
        _train_loop(spo, loader, predmodel,
                    lambda fn, cp, c, w, z: fn(cp, c, w, z))

    def test_blackbox(self):
        optmodel, loader = self._setup()
        predmodel = LinearPred(NUM_FEAT, optmodel.num_cost)
        dbb = pyepo.func.blackboxOpt(optmodel, lambd=10, processes=1)
        _train_loop(dbb, loader, predmodel,
                    lambda fn, cp, c, w, z: (fn(cp) * c).sum(1).mean())


# ============================================================
# TSP DFJ integration (exercises lazy-callback subtour-elim path)
# ============================================================

@requires_gurobi
class TestTspDFJIntegration:

    def _setup(self, num_nodes=5):
        from pyepo.data.dataset import optDataset
        from pyepo.model.grb.tsp import tspDFJModel
        x, c = pyepo.data.tsp.genData(NUM_DATA, NUM_FEAT, num_nodes, deg=1, seed=42)
        optmodel = tspDFJModel(num_nodes=num_nodes)
        dataset = optDataset(optmodel, x, c)
        loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)
        return optmodel, loader

    def test_blackbox(self):
        optmodel, loader = self._setup()
        predmodel = LinearPred(NUM_FEAT, optmodel.num_cost)
        dbb = pyepo.func.blackboxOpt(optmodel, lambd=10, processes=1)
        _train_loop(dbb, loader, predmodel,
                    lambda fn, cp, c, w, z: (fn(cp) * c).sum(1).mean())
