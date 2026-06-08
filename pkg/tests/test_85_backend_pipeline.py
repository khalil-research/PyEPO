#!/usr/bin/env python
"""Cross-backend pipeline: a representative predict -> solve -> loss -> metric ->
train slice run once per installed *non-Gurobi* backend (copt / ort / mpax).

Gurobi is exhaustively covered in test_30/50/60/80. This module instead verifies
the whole pipeline works on the other backends — most importantly MPAX's batched
dlpack bridge (optDataset + _solve_batch) and the MAXIMIZE sign flip, which a
Gurobi-only suite never reaches. Uninstalled backends skip via the fixture's
per-param marks; with none installed the tests are skipped entirely.
"""

import numpy as np
import torch

import pyepo
import pyepo.func as F

from .conftest import NUM_FEAT, LinearPred, take_batch

_STEPS = 2


class TestBackendPipeline:
    """Driven by the backend-parametrized sp_pipeline / ks_pipeline fixtures."""

    def test_dataset_objs_consistent(self, sp_pipeline):
        # objs == costs·sols confirms the solve (incl. the MPAX dlpack round-trip)
        # returns internally consistent tensors on this backend
        _optmodel, dataset, _loader = sp_pipeline
        recon = (dataset.costs * dataset.sols).sum(dim=1)
        np.testing.assert_allclose(recon.numpy(), dataset.objs.numpy().ravel(), atol=1e-2)

    def test_spoplus(self, sp_pipeline):
        optmodel, _ds, loader = sp_pipeline
        _x, c, w, z = take_batch(loader)
        cp = (c * 1.2).clone().detach().requires_grad_(True)
        loss = F.SPOPlus(optmodel, processes=1)(cp, c, w, z)
        assert torch.isfinite(loss).all()
        loss.backward()
        assert torch.isfinite(cp.grad).all()

    def test_perturbed_opt(self, sp_pipeline):
        optmodel, _ds, loader = sp_pipeline
        _x, c, _w, _z = take_batch(loader)
        cp = (c * 1.2).clone().detach().requires_grad_(True)
        out = F.DPO(optmodel, processes=1, n_samples=3)(cp)
        assert out.shape == cp.shape
        out.sum().backward()
        assert torch.isfinite(cp.grad).all()

    def test_perturbed_fenchel_young(self, sp_pipeline):
        optmodel, _ds, loader = sp_pipeline
        _x, c, w, _z = take_batch(loader)
        cp = (c * 1.2).clone().detach().requires_grad_(True)
        loss = F.PFY(optmodel, processes=1, n_samples=3)(cp, w)
        assert torch.isfinite(loss).all()
        loss.backward()
        assert torch.isfinite(cp.grad).all()

    def test_regret_metric(self, sp_pipeline):
        optmodel, _ds, loader = sp_pipeline
        pred = LinearPred(NUM_FEAT, optmodel.num_cost)
        reg = pyepo.metric.regret(pred, optmodel, loader)
        assert isinstance(reg, float) and reg >= 0

    def test_train_loop(self, sp_pipeline):
        optmodel, _ds, loader = sp_pipeline
        pred = LinearPred(NUM_FEAT, optmodel.num_cost)
        before = pred.linear.weight.detach().clone()
        spo = F.SPOPlus(optmodel, processes=1)
        opt = torch.optim.SGD(pred.parameters(), lr=1e-2)
        for i, (x, c, w, z) in enumerate(loader):
            if i >= _STEPS:
                break
            loss = spo(pred(x), c, w, z)
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert not torch.equal(before, pred.linear.weight.detach())

    def test_maximize_sign(self, ks_pipeline):
        # MAXIMIZE: objs match recon (sign flip correct) and SPO+ stays non-negative
        optmodel, dataset, loader = ks_pipeline
        recon = (dataset.costs * dataset.sols).sum(dim=1)
        np.testing.assert_allclose(recon.numpy(), dataset.objs.numpy().ravel(), atol=1e-2)
        assert (dataset.objs.numpy() >= -1e-3).all()
        _x, c, w, z = take_batch(loader)
        cp = (c * 1.2).clone().detach().requires_grad_(True)
        loss = F.SPOPlus(optmodel, processes=1)(cp, c, w, z)
        assert loss.item() >= -1e-6
        loss.backward()
        assert torch.isfinite(cp.grad).all()
