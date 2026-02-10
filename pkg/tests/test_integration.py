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
except (ImportError, NameError):
    _HAS_GUROBI = False

try:
    from pyepo.model.ort.shortestpath import shortestPathModel as ortSPModel
    from pyepo.model.ort.knapsack import knapsackModel as ortKnapsackModel
    ortSPModel(grid=(3, 3))
    _HAS_ORTOOLS = True
except (ImportError, NameError, Exception):
    _HAS_ORTOOLS = False

requires_gurobi = pytest.mark.skipif(not _HAS_GUROBI, reason="Gurobi not installed")
requires_ortools = pytest.mark.skipif(not _HAS_ORTOOLS, reason="OR-Tools not installed")


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


# ============================================================
# DYS-Net integration (uses OR-Tools, no Gurobi needed)
# ============================================================

def _sp_constraint_matrices(grid):
    """Build flow conservation constraint matrices for shortest path."""
    from pyepo.model.opt import _get_grid_arcs
    arcs = _get_grid_arcs(grid)
    num_nodes = grid[0] * grid[1]
    num_arcs = len(arcs)
    A = np.zeros((num_nodes, num_arcs), dtype=np.float32)
    for idx, (s, e) in enumerate(arcs):
        A[s, idx] = 1
        A[e, idx] = -1
    b = np.zeros(num_nodes, dtype=np.float32)
    b[0] = 1
    b[num_nodes - 1] = -1
    l = np.zeros(num_arcs, dtype=np.float32)
    u = np.ones(num_arcs, dtype=np.float32)
    return A, b, l, u


@requires_ortools
class TestDysOptIntegration:

    def test_train_shortest_path(self):
        """End-to-end training with DYS-Net on shortest path."""
        A, b, l, u = _sp_constraint_matrices(_GRID)
        num_cost = A.shape[1]
        dys = pyepo.func.dysOpt(A, b, l, u,
                                alpha=0.1, max_iter=500, tol=1e-3)
        predmodel = _LinearModel(_NUM_FEAT, num_cost)
        optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-2)

        # generate synthetic data
        np.random.seed(42)
        x_data = np.random.randn(_NUM_DATA, _NUM_FEAT).astype(np.float32)
        c_data = np.abs(np.random.randn(_NUM_DATA, num_cost)).astype(np.float32)

        x_tensor = torch.tensor(x_data)
        c_tensor = torch.tensor(c_data)

        dys.train()
        losses = []
        for step in range(_STEPS):
            cp = predmodel(x_tensor[:_BATCH])
            sol = dys(cp)
            # task loss: predicted cost on predicted solution
            loss = torch.mean(torch.sum(c_tensor[:_BATCH] * sol, dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # verify loss is finite
        assert all(np.isfinite(l) for l in losses)

    def test_solution_quality_vs_solver(self):
        """DYS solutions should have reasonable objective compared to solver."""
        optmodel = ortSPModel(grid=_GRID)
        A, b, l, u = _sp_constraint_matrices(_GRID)
        dys = pyepo.func.dysOpt(A, b, l, u,
                                alpha=0.1, max_iter=1000, tol=1e-4)
        dys.eval()

        np.random.seed(123)
        costs = np.random.rand(4, optmodel.num_cost).astype(np.float32)
        pred_cost = torch.tensor(costs)

        with torch.no_grad():
            dys_sols = dys(pred_cost).numpy()

        for i in range(4):
            # exact solver
            optmodel.setObj(costs[i])
            _, exact_obj = optmodel.solve()
            dys_obj = costs[i] @ dys_sols[i]
            # DYS objective should not be more than 20% worse
            gap = abs(dys_obj - exact_obj) / (abs(exact_obj) + 1e-8)
            assert gap < 0.2, f"Instance {i}: gap {gap:.2%} too large"

    def test_maximize_knapsack(self):
        """DYS-Net with minimize=False (maximization)."""
        n = 4
        # equality: x_0 = 0.5
        A = np.zeros((1, n), dtype=np.float32)
        A[0, 0] = 1.0
        b = np.array([0.5], dtype=np.float32)
        l = np.zeros(n, dtype=np.float32)
        u = np.ones(n, dtype=np.float32)

        dys = pyepo.func.dysOpt(A, b, l, u,
                                alpha=0.1, max_iter=500, tol=1e-3, minimize=False)
        predmodel = _LinearModel(_NUM_FEAT, n)
        optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-2)

        dys.train()
        x_data = torch.randn(_BATCH, _NUM_FEAT)
        for _ in range(_STEPS):
            cp = predmodel(x_data)
            sol = dys(cp)
            loss = -torch.mean(torch.sum(cp * sol, dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
