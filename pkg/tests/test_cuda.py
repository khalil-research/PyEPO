#!/usr/bin/env python
# coding: utf-8
"""
CUDA device tests: verify tensors stay on the correct device throughout
the predict-then-optimize pipeline (forward, loss, backward, metrics).

Skipped automatically when CUDA is not available.

Shared fixtures (sp_data, ks_data, LinearPred, constants) live in conftest.py.
"""

import pytest
import torch

import pyepo

from .conftest import LinearPred, NUM_FEAT, _HAS_GUROBI

_HAS_CUDA = torch.cuda.is_available()
_DEVICE = torch.device("cuda" if _HAS_CUDA else "cpu")
_STEPS = 2

requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")
requires_cuda_gurobi = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_GUROBI),
    reason="CUDA or Gurobi not available"
)


# ============================================================
# Helpers
# ============================================================

def _assert_cuda(tensor, name="tensor"):
    """Assert a tensor is on CUDA."""
    assert tensor.device.type == "cuda", f"{name} not on CUDA: {tensor.device}"


def _assert_grads_cuda(model):
    """Assert all parameter gradients exist and are on CUDA."""
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        _assert_cuda(param.grad, f"grad of {name}")


def _get_batch(loader):
    """Get one batch and move to CUDA."""
    x, c, w, z = next(iter(loader))
    return x.to(_DEVICE), c.to(_DEVICE), w.to(_DEVICE), z.to(_DEVICE)


def _fresh_predmodel(num_cost):
    return LinearPred(NUM_FEAT, num_cost).to(_DEVICE)


# ============================================================
# Model parameter device
# ============================================================

@requires_cuda
class TestModelDevice:

    def test_parameters_on_cuda(self):
        pred = LinearPred(5, 10).to(_DEVICE)
        for name, param in pred.named_parameters():
            _assert_cuda(param, name)

    def test_forward_output_on_cuda(self):
        pred = LinearPred(5, 10).to(_DEVICE)
        x = torch.randn(4, 5, device=_DEVICE)
        out = pred(x)
        _assert_cuda(out, "forward output")


# ============================================================
# Surrogate losses: SPOPlus, perturbationGradient
# ============================================================

@requires_cuda_gurobi
class TestSPOPlusCUDA:

    def test_loss_and_grad_on_cuda(self, sp_data):
        optmodel, dataset, loader = sp_data
        x, c, w, z = _get_batch(loader)
        pred = _fresh_predmodel(optmodel.num_cost)
        cp = pred(x)
        _assert_cuda(cp, "cp")
        spo = pyepo.func.SPOPlus(optmodel, processes=1)
        loss = spo(cp, c, w, z)
        _assert_cuda(loss, "SPOPlus loss")
        loss.backward()
        _assert_grads_cuda(pred)

    def test_train_loop(self, sp_data):
        optmodel, dataset, loader = sp_data
        pred = _fresh_predmodel(optmodel.num_cost)
        opt = torch.optim.SGD(pred.parameters(), lr=1e-2)
        spo = pyepo.func.SPOPlus(optmodel, processes=1)
        for i, (x, c, w, z) in enumerate(loader):
            if i >= _STEPS:
                break
            x, c, w, z = x.to(_DEVICE), c.to(_DEVICE), w.to(_DEVICE), z.to(_DEVICE)
            loss = spo(pred(x), c, w, z)
            opt.zero_grad()
            loss.backward()
            opt.step()


@requires_cuda_gurobi
class TestPerturbationGradientCUDA:

    def test_loss_and_grad_on_cuda(self, sp_data):
        optmodel, dataset, loader = sp_data
        x, c, w, z = _get_batch(loader)
        pred = _fresh_predmodel(optmodel.num_cost)
        cp = pred(x)
        pg = pyepo.func.perturbationGradient(optmodel, processes=1, sigma=1.0)
        loss = pg(cp, c)
        _assert_cuda(loss, "perturbationGradient loss")
        loss.backward()
        _assert_grads_cuda(pred)


# ============================================================
# Blackbox losses: blackboxOpt, negativeIdentity
# ============================================================

@requires_cuda_gurobi
class TestBlackboxOptCUDA:

    def test_output_and_grad_on_cuda(self, sp_data):
        optmodel, dataset, loader = sp_data
        x, c, w, z = _get_batch(loader)
        pred = _fresh_predmodel(optmodel.num_cost)
        cp = pred(x)
        bb = pyepo.func.blackboxOpt(optmodel, processes=1, lambd=10)
        w_hat = bb(cp)
        _assert_cuda(w_hat, "blackboxOpt output")
        loss = w_hat.mean()
        loss.backward()
        _assert_grads_cuda(pred)


@requires_cuda_gurobi
class TestNegativeIdentityCUDA:

    def test_output_and_grad_on_cuda(self, sp_data):
        optmodel, dataset, loader = sp_data
        x, c, w, z = _get_batch(loader)
        pred = _fresh_predmodel(optmodel.num_cost)
        cp = pred(x)
        nid = pyepo.func.negativeIdentity(optmodel, processes=1)
        w_hat = nid(cp)
        _assert_cuda(w_hat, "negativeIdentity output")
        loss = w_hat.mean()
        loss.backward()
        _assert_grads_cuda(pred)


# ============================================================
# Perturbed losses: perturbedOpt, perturbedFenchelYoung,
#                   implicitMLE, adaptiveImplicitMLE
# ============================================================

@requires_cuda_gurobi
class TestPerturbedOptCUDA:

    def test_output_and_grad_on_cuda(self, sp_data):
        optmodel, dataset, loader = sp_data
        x, c, w, z = _get_batch(loader)
        pred = _fresh_predmodel(optmodel.num_cost)
        cp = pred(x)
        ptb = pyepo.func.perturbedOpt(optmodel, processes=1, n_samples=3, sigma=1.0)
        w_hat = ptb(cp)
        _assert_cuda(w_hat, "perturbedOpt output")
        loss = w_hat.mean()
        loss.backward()
        _assert_grads_cuda(pred)


@requires_cuda_gurobi
class TestPerturbedFenchelYoungCUDA:

    def test_loss_and_grad_on_cuda(self, sp_data):
        optmodel, dataset, loader = sp_data
        x, c, w, z = _get_batch(loader)
        pred = _fresh_predmodel(optmodel.num_cost)
        cp = pred(x)
        pfy = pyepo.func.perturbedFenchelYoung(optmodel, processes=1, n_samples=3, sigma=1.0)
        loss = pfy(cp, w)
        _assert_cuda(loss, "perturbedFenchelYoung loss")
        loss.backward()
        _assert_grads_cuda(pred)


@requires_cuda_gurobi
class TestImplicitMLECUDA:

    def test_output_and_grad_on_cuda(self, sp_data):
        optmodel, dataset, loader = sp_data
        x, c, w, z = _get_batch(loader)
        pred = _fresh_predmodel(optmodel.num_cost)
        cp = pred(x)
        imle = pyepo.func.implicitMLE(optmodel, processes=1, n_samples=3, sigma=1.0)
        w_hat = imle(cp)
        _assert_cuda(w_hat, "implicitMLE output")
        loss = w_hat.mean()
        loss.backward()
        _assert_grads_cuda(pred)


@requires_cuda_gurobi
class TestAdaptiveImplicitMLECUDA:

    def test_output_and_grad_on_cuda(self, sp_data):
        optmodel, dataset, loader = sp_data
        x, c, w, z = _get_batch(loader)
        pred = _fresh_predmodel(optmodel.num_cost)
        cp = pred(x)
        aimle = pyepo.func.adaptiveImplicitMLE(optmodel, processes=1, n_samples=3, sigma=1.0)
        w_hat = aimle(cp)
        _assert_cuda(w_hat, "adaptiveImplicitMLE output")
        loss = w_hat.mean()
        loss.backward()
        _assert_grads_cuda(pred)


# ============================================================
# Contrastive losses: NCE, contrastiveMAP
# ============================================================

@requires_cuda_gurobi
class TestNCECUDA:

    def test_loss_and_grad_on_cuda(self, sp_data):
        optmodel, dataset, loader = sp_data
        x, c, w, z = _get_batch(loader)
        pred = _fresh_predmodel(optmodel.num_cost)
        cp = pred(x)
        nce = pyepo.func.NCE(optmodel, processes=1, solve_ratio=1, dataset=dataset)
        loss = nce(cp, w)
        _assert_cuda(loss, "NCE loss")
        loss.backward()
        _assert_grads_cuda(pred)


@requires_cuda_gurobi
class TestContrastiveMAPCUDA:

    def test_loss_and_grad_on_cuda(self, sp_data):
        optmodel, dataset, loader = sp_data
        x, c, w, z = _get_batch(loader)
        pred = _fresh_predmodel(optmodel.num_cost)
        cp = pred(x)
        cmap = pyepo.func.contrastiveMAP(optmodel, processes=1, solve_ratio=1, dataset=dataset)
        loss = cmap(cp, w)
        _assert_cuda(loss, "contrastiveMAP loss")
        loss.backward()
        _assert_grads_cuda(pred)


# ============================================================
# Ranking losses: listwiseLTR, pairwiseLTR, pointwiseLTR
# ============================================================

@requires_cuda_gurobi
class TestListwiseLTRCUDA:

    def test_loss_and_grad_on_cuda(self, sp_data):
        optmodel, dataset, loader = sp_data
        x, c, w, z = _get_batch(loader)
        pred = _fresh_predmodel(optmodel.num_cost)
        cp = pred(x)
        lw = pyepo.func.listwiseLTR(optmodel, processes=1, solve_ratio=1, dataset=dataset)
        loss = lw(cp, c)
        _assert_cuda(loss, "listwiseLTR loss")
        loss.backward()
        _assert_grads_cuda(pred)


@requires_cuda_gurobi
class TestPairwiseLTRCUDA:

    def test_loss_and_grad_on_cuda(self, sp_data):
        optmodel, dataset, loader = sp_data
        x, c, w, z = _get_batch(loader)
        pred = _fresh_predmodel(optmodel.num_cost)
        cp = pred(x)
        pw = pyepo.func.pairwiseLTR(optmodel, processes=1, solve_ratio=1, dataset=dataset)
        loss = pw(cp, c)
        _assert_cuda(loss, "pairwiseLTR loss")
        loss.backward()
        _assert_grads_cuda(pred)


@requires_cuda_gurobi
class TestPointwiseLTRCUDA:

    def test_loss_and_grad_on_cuda(self, sp_data):
        optmodel, dataset, loader = sp_data
        x, c, w, z = _get_batch(loader)
        pred = _fresh_predmodel(optmodel.num_cost)
        cp = pred(x)
        pt = pyepo.func.pointwiseLTR(optmodel, processes=1, solve_ratio=1, dataset=dataset)
        loss = pt(cp, c)
        _assert_cuda(loss, "pointwiseLTR loss")
        loss.backward()
        _assert_grads_cuda(pred)


# ============================================================
# MAXIMIZE problem (knapsack) on CUDA
# ============================================================

@requires_cuda_gurobi
class TestKnapsackCUDA:

    def test_spo_plus(self, ks_data):
        optmodel, dataset, loader = ks_data
        x, c, w, z = _get_batch(loader)
        pred = _fresh_predmodel(optmodel.num_cost)
        cp = pred(x)
        spo = pyepo.func.SPOPlus(optmodel, processes=1)
        loss = spo(cp, c, w, z)
        _assert_cuda(loss, "knapsack SPOPlus loss")
        loss.backward()
        _assert_grads_cuda(pred)

    def test_blackbox_opt(self, ks_data):
        optmodel, dataset, loader = ks_data
        x, c, w, z = _get_batch(loader)
        pred = _fresh_predmodel(optmodel.num_cost)
        cp = pred(x)
        bb = pyepo.func.blackboxOpt(optmodel, processes=1, lambd=10)
        w_hat = bb(cp)
        _assert_cuda(w_hat, "knapsack blackboxOpt output")
        loss = -(c * w_hat).sum(dim=1).mean()
        loss.backward()
        _assert_grads_cuda(pred)


# ============================================================
# Metrics with CUDA model
# ============================================================

@requires_cuda_gurobi
class TestMetricsCUDA:

    def test_regret(self, sp_data):
        optmodel, dataset, loader = sp_data
        pred = _fresh_predmodel(optmodel.num_cost)
        reg = pyepo.metric.regret(pred, optmodel, loader)
        assert isinstance(reg, float)
        assert reg >= 0

    def test_mse(self, sp_data):
        optmodel, dataset, loader = sp_data
        pred = _fresh_predmodel(optmodel.num_cost)
        mse = pyepo.metric.MSE(pred, loader)
        assert isinstance(mse, float)
        assert mse >= 0

    def test_unambiguous_regret(self, sp_data):
        optmodel, dataset, loader = sp_data
        pred = _fresh_predmodel(optmodel.num_cost)
        unamb = pyepo.metric.unambRegret(pred, optmodel, loader)
        assert isinstance(unamb, float)

    def test_regret_knapsack(self, ks_data):
        optmodel, dataset, loader = ks_data
        pred = _fresh_predmodel(optmodel.num_cost)
        reg = pyepo.metric.regret(pred, optmodel, loader)
        assert isinstance(reg, float)
        assert reg >= 0
