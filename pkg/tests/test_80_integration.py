#!/usr/bin/env python
"""End-to-end integration: a few real training steps through the full pipeline.

Where test_50_func checks each loss's forward/backward in isolation, this layer
drives the whole loop (predict -> loss -> backward -> optimizer.step) over
multiple batches, plus the dataset variants that only matter end to end (kNN,
CaVE binding constraints, portfolio QP, TSP-DFJ lazy cuts) and the MAXIMIZE
case. The multiprocessing path is exercised once, marked ``slow`` (Windows
process spawn is the suite's main time sink).
"""

import sys

import pytest
import torch
from torch.utils.data import DataLoader

import pyepo
import pyepo.func as F
from pyepo.data.dataset import optDataset, optDatasetKNN

from .conftest import (
    BATCH,
    GRID,
    LOSS_REGISTRY,
    NUM_DATA,
    NUM_FEAT,
    LinearPred,
    call_op,
    requires_gurobi,
)

_STEPS = 2


def _train_loop(loss_fn, loader, predmodel, call, steps=_STEPS):
    opt = torch.optim.SGD(predmodel.parameters(), lr=1e-2)
    for i, (x, c, w, z) in enumerate(loader):
        if i >= steps:
            break
        cp = predmodel(x)
        loss = call(loss_fn, cp, c, w, z)
        opt.zero_grad()
        loss.backward()
        opt.step()


def _scalar_loss_call(name):
    """Adapt a registry op to a scalar training loss: loss-ops return their
    scalar directly; solution-ops are paired with the decision loss c·w."""
    kind, _build, sig = LOSS_REGISTRY[name]
    if kind == "loss":
        return lambda fn, cp, c, w, z: call_op(fn, sig, cp, c, w, z)
    return lambda fn, cp, c, w, z: (call_op(fn, sig, cp, c, w, z) * c).sum(1).mean()


# Per-loss forward/backward is gated exhaustively in test_50; the e2e loop only
# adds the optimizer.step + multi-batch path, which is identical across losses.
# Sample one loss per mechanism: the three scalar-loss sigs (cp,c,w,z / cp,c /
# cp,w), the blackbox / perturbed / Frank-Wolfe solution-ops, and one pool loss
# (NCE) whose cached solution pool grows across batches.
_E2E_LOSSES = ["SPOPlus", "PG", "PFY", "DBB", "DPO", "RFWO", "NCE"]


@requires_gurobi
@pytest.mark.parametrize("name", _E2E_LOSSES)
def test_train_minimize(name, sp_data):
    optmodel, dataset, loader = sp_data
    build = LOSS_REGISTRY[name][1]
    predmodel = LinearPred(NUM_FEAT, optmodel.num_cost)
    before = predmodel.linear.weight.detach().clone()
    _train_loop(build(optmodel, dataset, "mean"), loader, predmodel, _scalar_loss_call(name))
    # an optimizer step actually moved the parameters
    assert not torch.equal(before, predmodel.linear.weight.detach())


@requires_gurobi
class TestMaximizeEndToEnd:
    def test_spo_plus_knapsack(self, ks_data):
        optmodel, _ds, loader = ks_data
        predmodel = LinearPred(NUM_FEAT, optmodel.num_cost)
        _train_loop(
            F.SPOPlus(optmodel, processes=1),
            loader,
            predmodel,
            lambda fn, cp, c, w, z: fn(cp, c, w, z),
        )

    def test_blackbox_knapsack(self, ks_data):
        optmodel, _ds, loader = ks_data
        predmodel = LinearPred(NUM_FEAT, optmodel.num_cost)
        _train_loop(
            F.DBB(optmodel, lambd=10, processes=1),
            loader,
            predmodel,
            lambda fn, cp, c, w, z: -(fn(cp) * c).sum(1).mean(),
        )


@requires_gurobi
class TestSpecialDatasets:
    def test_knn_dataset(self):
        from pyepo.model.grb.shortestpath import shortestPathModel

        x, c = pyepo.data.shortestpath.genData(NUM_DATA, NUM_FEAT, GRID, seed=42)
        optmodel = shortestPathModel(grid=GRID)
        dataset = optDatasetKNN(optmodel, x, c, k=3, weight=0.5)
        assert len(dataset) == NUM_DATA
        loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)
        predmodel = LinearPred(NUM_FEAT, optmodel.num_cost)
        _train_loop(
            F.SPOPlus(optmodel, processes=1),
            loader,
            predmodel,
            lambda fn, cp, c_b, w, z: fn(cp, c_b, w, z),
        )

    def test_portfolio_qp(self):
        from pyepo.model.grb.portfolio import portfolioModel

        cov, x, c = pyepo.data.portfolio.genData(NUM_DATA, NUM_FEAT, 8, deg=1, seed=42)
        optmodel = portfolioModel(num_assets=8, covariance=cov)
        loader = DataLoader(optDataset(optmodel, x, c), batch_size=BATCH, shuffle=False)
        predmodel = LinearPred(NUM_FEAT, optmodel.num_cost)
        _train_loop(
            F.SPOPlus(optmodel, processes=1),
            loader,
            predmodel,
            lambda fn, cp, c_b, w, z: fn(cp, c_b, w, z),
        )

    def test_tsp_dfj_lazy_callback(self):
        from pyepo.model.grb.tsp import tspDFJModel

        x, c = pyepo.data.tsp.genData(NUM_DATA, NUM_FEAT, 5, deg=1, seed=42)
        optmodel = tspDFJModel(num_nodes=5)
        loader = DataLoader(optDataset(optmodel, x, c), batch_size=BATCH, shuffle=False)
        predmodel = LinearPred(NUM_FEAT, optmodel.num_cost)
        _train_loop(
            F.DBB(optmodel, lambd=10, processes=1),
            loader,
            predmodel,
            lambda fn, cp, c_b, w, z: (fn(cp) * c_b).sum(1).mean(),
        )


@requires_gurobi
class TestCaVEEndToEnd:
    """CaVE on shortest path: optDatasetConstrs + collated loader + cone loss."""

    def _train(self, loss_fn, loader, predmodel):
        opt = torch.optim.SGD(predmodel.parameters(), lr=1e-2)
        for i, (x, _c, _w, _z, ctrs) in enumerate(loader):
            if i >= _STEPS:
                break
            loss = loss_fn(predmodel(x), ctrs)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def test_default_preset(self, sp_constrs_data):
        optmodel, _ds, loader = sp_constrs_data
        predmodel = LinearPred(NUM_FEAT, optmodel.num_cost)
        self._train(F.CaVE(optmodel, processes=1), loader, predmodel)


@requires_gurobi
@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="pathos spawn-based pool segfaults at interpreter teardown once JAX is "
    "loaded in-process on Windows; the multiprocessing path is validated on "
    "Linux CI (fork)",
)
class TestParallelSolving:
    """Multiprocessing solve path (processes>1). Marked slow: process spawn
    dominates wall-clock; deselect with -m 'not slow'."""

    def test_spo_plus_parallel(self, sp_data):
        optmodel, _ds, loader = sp_data
        predmodel = LinearPred(NUM_FEAT, optmodel.num_cost)
        _train_loop(
            F.SPOPlus(optmodel, processes=2),
            loader,
            predmodel,
            lambda fn, cp, c, w, z: fn(cp, c, w, z),
        )
