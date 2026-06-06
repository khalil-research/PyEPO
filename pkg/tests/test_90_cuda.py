#!/usr/bin/env python
"""CUDA device tests: tensors and gradients stay on GPU through the pipeline.

Skipped automatically when CUDA is unavailable. The optimization solve happens
on CPU inside each loss; these tests assert the autograd tensors entering and
leaving the loss remain on the GPU and that gradients land on GPU parameters.
"""

import pytest
import torch

import pyepo
import pyepo.func as F

from .conftest import (
    _HAS_CUDA,
    _HAS_GUROBI,
    LOSS_OPS,
    LOSS_REGISTRY,
    NUM_FEAT,
    SOLUTION_OPS,
    LinearPred,
    call_op,
    requires_cuda,
    requires_jax_gpu,
)

_DEVICE = torch.device("cuda" if _HAS_CUDA else "cpu")

requires_cuda_gurobi = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_GUROBI), reason="CUDA or Gurobi not available")


def _cuda_batch(loader, n=4):
    x, c, w, z = next(iter(loader))
    return (x[:n].to(_DEVICE), c[:n].to(_DEVICE), w[:n].to(_DEVICE), z[:n].to(_DEVICE))


def _assert_cuda(t, name):
    assert t.device.type == "cuda", f"{name} on {t.device}"


def _assert_grads_cuda(model):
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        _assert_cuda(p.grad, f"grad {name}")


@requires_cuda
class TestModelDevice:

    def test_parameters_and_forward_on_cuda(self):
        pred = LinearPred(5, 10).to(_DEVICE)
        for name, p in pred.named_parameters():
            _assert_cuda(p, name)
        out = pred(torch.randn(4, 5, device=_DEVICE))
        _assert_cuda(out, "forward output")


@requires_cuda_gurobi
class TestSolutionLossesCUDA:

    @pytest.mark.parametrize("name", SOLUTION_OPS)
    def test_output_and_grad_on_cuda(self, name, sp_data):
        optmodel, dataset, loader = sp_data
        _kind, build, sig = LOSS_REGISTRY[name]
        x, c, w, z = _cuda_batch(loader)
        pred = LinearPred(NUM_FEAT, optmodel.num_cost).to(_DEVICE)
        out = call_op(build(optmodel, dataset, "mean"), sig, pred(x), c, w, z)
        _assert_cuda(out, "output")
        out.mean().backward()
        _assert_grads_cuda(pred)


@requires_cuda_gurobi
class TestLossesCUDA:

    @pytest.mark.parametrize("name", LOSS_OPS)
    def test_loss_and_grad_on_cuda(self, name, sp_data):
        optmodel, dataset, loader = sp_data
        _kind, build, sig = LOSS_REGISTRY[name]
        x, c, w, z = _cuda_batch(loader)
        pred = LinearPred(NUM_FEAT, optmodel.num_cost).to(_DEVICE)
        loss = call_op(build(optmodel, dataset, "mean"), sig, pred(x), c, w, z)
        _assert_cuda(loss, "loss")
        loss.backward()
        _assert_grads_cuda(pred)


@requires_cuda_gurobi
class TestMaximizeCUDA:

    def test_spo_plus_knapsack(self, ks_data):
        optmodel, _ds, loader = ks_data
        x, c, w, z = _cuda_batch(loader)
        pred = LinearPred(NUM_FEAT, optmodel.num_cost).to(_DEVICE)
        loss = F.SPOPlus(optmodel, processes=1)(pred(x), c, w, z)
        _assert_cuda(loss, "knapsack SPO+ loss")
        loss.backward()
        _assert_grads_cuda(pred)


@requires_cuda_gurobi
class TestMetricsCUDA:

    def test_regret_and_mse(self, sp_data):
        optmodel, _ds, loader = sp_data
        pred = LinearPred(NUM_FEAT, optmodel.num_cost).to(_DEVICE)
        assert pyepo.metric.regret(pred, optmodel, loader) >= 0
        assert pyepo.metric.MSE(pred, loader) >= 0

    def test_unamb_regret(self, sp_data):
        optmodel, _ds, loader = sp_data
        pred = LinearPred(NUM_FEAT, optmodel.num_cost).to(_DEVICE)
        assert isinstance(pyepo.metric.unambRegret(pred, optmodel, loader), float)


@requires_jax_gpu
class TestMpaxGpuBridge:
    """jax-gpu <-> torch-cuda dlpack path: an MPAX solve on a GPU jax device must
    hand tensors back to torch on CUDA. Double-gated (CUDA + jax-gpu); skips on
    CPU-only jax, but keeps the GPU bridge under test wherever both are present
    instead of relying on manual verification."""

    def test_spoplus_loss_and_grad_on_cuda(self):
        from pyepo.data.shortestpath import genData
        from pyepo.model.mpax.shortestpath import shortestPathModel
        optmodel = shortestPathModel(grid=(3, 3))
        x, c = genData(8, NUM_FEAT, (3, 3), seed=42)
        pred = LinearPred(NUM_FEAT, optmodel.num_cost).to(_DEVICE)
        xb = torch.as_tensor(x, dtype=torch.float32, device=_DEVICE)
        cb = torch.as_tensor(c, dtype=torch.float32, device=_DEVICE)
        cp = pred(xb)
        _assert_cuda(cp, "cp")
        # SPO+ solves through MPAX on the GPU jax device, returns via dlpack
        w = torch.zeros_like(cb)
        z = torch.zeros(cb.shape[0], 1, device=_DEVICE)
        loss = F.SPOPlus(optmodel, processes=1)(cp, cb, w, z)
        _assert_cuda(loss, "MPAX-GPU SPO+ loss")
        loss.backward()
        _assert_grads_cuda(pred)
