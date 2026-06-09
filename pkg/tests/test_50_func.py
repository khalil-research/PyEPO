#!/usr/bin/env python
"""Tests for pyepo.func: autograd losses, helpers, and the optModule base.

Three groups:

1. Pure / mock helpers (solution pool, cache selection, sum-of-gamma noise,
   solution check) and optModule init validation — no solver.
2. A focused forward/backward contract applied to every loss via parametrized
   registries: solution-returning ops must return (batch, vars) with finite
   gradients; loss-returning ops must return a scalar that backpropagates, and
   honour reduction = mean/sum/none.
3. Deep correctness gates that compare an autograd gradient to its closed form
   or a finite-difference Jacobian (SPO+, regularized Frank-Wolfe).
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from pyepo import EPO
from pyepo.func.utils import (
    _cache_in_pass,
    _check_sol,
    _update_solution_pool,
    sumGammaDistribution,
)

from .conftest import (
    FD_TRUTH,
    LOSS_OPS,
    LOSS_REGISTRY,
    SOLUTION_OPS,
    call_op,
    finite_diff_grad,
    requires_clarabel,
    requires_gurobi,
    take_batch,
)

# ============================================================
# 1a. Pure helpers
# ============================================================


class TestCacheInPass:
    def _mock_model(self, sense):
        m = MagicMock()
        m.modelSense = sense
        return m

    def test_minimize_selects_min_obj(self):
        cp = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        solpool = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        sol, _obj, _ = _cache_in_pass(cp, self._mock_model(EPO.MINIMIZE), solpool)
        assert torch.allclose(sol[0], solpool[0])
        assert torch.allclose(sol[1], solpool[1])

    def test_maximize_selects_max_obj(self):
        cp = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        solpool = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        sol, _obj, _ = _cache_in_pass(cp, self._mock_model(EPO.MAXIMIZE), solpool)
        assert torch.allclose(sol[0], solpool[1])
        assert torch.allclose(sol[1], solpool[0])

    def test_invalid_sense_raises(self):
        m = MagicMock()
        m.modelSense = 999
        with pytest.raises(ValueError):
            _cache_in_pass(torch.tensor([[1.0, 2.0]]), m, torch.tensor([[1.0, 0.0]]))

    def test_output_shapes(self):
        cp = torch.randn(5, 4)
        solpool = torch.randn(10, 4)
        sol, obj, _ = _cache_in_pass(cp, self._mock_model(EPO.MINIMIZE), solpool)
        assert sol.shape == (5, 4)
        assert obj.shape == (5,)


class TestCheckSol:
    def test_correct_passes(self):
        _check_sol(
            torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[1.0, 0.0, 1.0]]), torch.tensor([4.0])
        )

    def test_incorrect_raises(self):
        with pytest.raises(AssertionError):
            _check_sol(
                torch.tensor([[1.0, 2.0, 3.0]]),
                torch.tensor([[1.0, 0.0, 1.0]]),
                torch.tensor([999.0]),
            )


class TestSumGammaDistribution:
    def test_sample_shape(self):
        dist = sumGammaDistribution(kappa=1.0, n_iterations=10, seed=42)
        assert dist.sample((5, 3)).shape == (5, 3)

    def test_deterministic(self):
        s1 = sumGammaDistribution(kappa=1.0, seed=42).sample((10,))
        s2 = sumGammaDistribution(kappa=1.0, seed=42).sample((10,))
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds(self):
        s1 = sumGammaDistribution(kappa=1.0, seed=0).sample((100,))
        s2 = sumGammaDistribution(kappa=1.0, seed=1).sample((100,))
        assert not np.array_equal(s1, s2)

    def test_finite(self):
        s = sumGammaDistribution(kappa=2.0, n_iterations=20, seed=42).sample((100,))
        assert np.all(np.isfinite(s))


class TestSolutionPool:
    def test_init_pool(self):
        pool = _update_solution_pool(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), None)
        assert pool.shape == (2, 2)

    def test_dedup_identical(self):
        sol = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        pool = _update_solution_pool(sol, None)
        pool = _update_solution_pool(sol, pool)
        assert pool.shape[0] == 2

    def test_append_new(self):
        pool = _update_solution_pool(torch.tensor([[1.0, 0.0]]), None)
        pool = _update_solution_pool(torch.tensor([[0.0, 1.0]]), pool)
        assert pool.shape[0] == 2

    def test_mixed_new_and_dup(self):
        pool = _update_solution_pool(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), None)
        pool = _update_solution_pool(torch.tensor([[1.0, 0.0], [1.0, 1.0]]), pool)
        assert pool.shape[0] == 3


# ============================================================
# 1b. optModule init validation
# ============================================================


@requires_gurobi
class TestOptModuleInit:
    def _model(self):
        from pyepo.model.grb.shortestpath import shortestPathModel

        return shortestPathModel(grid=(3, 3))

    def test_invalid_model_type_raises(self):
        from pyepo.func.surrogate import SPOPlus

        with pytest.raises(TypeError):
            SPOPlus("not_a_model")

    def test_invalid_processes_raises(self):
        from pyepo.func.surrogate import SPOPlus

        with pytest.raises(ValueError):
            SPOPlus(self._model(), processes=-1)

    @pytest.mark.parametrize("ratio", [1.5, -0.1])
    def test_invalid_solve_ratio_raises(self, ratio):
        from pyepo.func.surrogate import SPOPlus

        with pytest.raises(ValueError):
            SPOPlus(self._model(), solve_ratio=ratio)

    def test_solve_ratio_lt1_requires_dataset(self):
        from pyepo.func.surrogate import SPOPlus

        with pytest.raises(TypeError):
            SPOPlus(self._model(), solve_ratio=0.5, dataset=None)


# ============================================================
# 2. Forward/backward contract for every loss
# ============================================================
# Losses are constructed from the shared conftest.LOSS_REGISTRY; here we only
# assert the forward/backward contract. Solution-returning ops yield (batch,
# vars); loss-returning ops yield a scalar and honour reduction = mean/sum/none.


@requires_gurobi
class TestSolutionLossContract:
    """Ops that return a predicted solution of shape (batch, vars)."""

    @pytest.mark.parametrize("name", SOLUTION_OPS)
    def test_forward_shape_and_backward(self, name, sp_data):
        optmodel, dataset, loader = sp_data
        _kind, build, sig = LOSS_REGISTRY[name]
        _x, c, w, z = take_batch(loader)
        cp = (c * 1.2).clone().detach().requires_grad_(True)
        out = call_op(build(optmodel, dataset, "mean"), sig, cp, c, w, z)
        assert out.shape == cp.shape
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert cp.grad is not None
        assert cp.grad.shape == cp.shape
        assert torch.isfinite(cp.grad).all()


@requires_gurobi
class TestLossContract:
    """Ops that return a scalar differentiable loss."""

    @pytest.mark.parametrize("name", LOSS_OPS)
    def test_scalar_forward_and_backward(self, name, sp_data):
        optmodel, dataset, loader = sp_data
        _kind, build, sig = LOSS_REGISTRY[name]
        _x, c, w, z = take_batch(loader)
        cp = (c * 1.2).clone().detach().requires_grad_(True)
        loss = call_op(build(optmodel, dataset, "mean"), sig, cp, c, w, z)
        assert loss.dim() == 0
        loss.backward()
        assert cp.grad is not None and cp.grad.shape == cp.shape
        assert torch.isfinite(cp.grad).all()

    @pytest.mark.parametrize("name", LOSS_OPS)
    def test_reduction_modes(self, name, sp_data):
        optmodel, dataset, loader = sp_data
        _kind, build, sig = LOSS_REGISTRY[name]
        _x, c, w, z = take_batch(loader)
        cp = (c * 1.2).clone().detach()
        none = call_op(build(optmodel, dataset, "none"), sig, cp, c, w, z)
        # most losses reduce to (batch,); lsLTR keeps a (batch, pool) grid
        assert none.shape[0] == cp.shape[0]
        mean = call_op(build(optmodel, dataset, "mean"), sig, cp, c, w, z)
        total = call_op(build(optmodel, dataset, "sum"), sig, cp, c, w, z)
        np.testing.assert_allclose(mean.item(), none.mean().item(), atol=1e-5)
        np.testing.assert_allclose(total.item(), none.sum().item(), atol=1e-5)


@requires_gurobi
class TestMaximizeSense:
    """Losses must handle a MAXIMIZE problem (knapsack); SPO+ stays non-negative."""

    def test_spoplus_nonnegative(self, ks_data):
        from pyepo.func.surrogate import SPOPlus

        optmodel, _dataset, loader = ks_data
        _x, c, w, z = take_batch(loader)
        cp = (c * 1.2).clone().detach().requires_grad_(True)
        loss = SPOPlus(optmodel, processes=1)(cp, c, w, z)
        assert loss.item() >= -1e-6
        loss.backward()
        assert torch.isfinite(cp.grad).all()

    def test_perturbed_opt_runs(self, ks_data):
        from pyepo.func.perturbed import DPO

        optmodel, _dataset, loader = ks_data
        _x, c, _w, _z = take_batch(loader)
        cp = (c * 1.2).clone().detach().requires_grad_(True)
        out = DPO(optmodel, processes=1, n_samples=3)(cp)
        assert out.shape == cp.shape
        out.sum().backward()
        assert torch.isfinite(cp.grad).all()


@requires_gurobi
class TestSolveRatioCaching:
    """solve_ratio < 1 routes some passes through the cached solution pool."""

    def test_spoplus_with_caching_runs(self, sp_data):
        from pyepo.func.surrogate import SPOPlus

        optmodel, dataset, loader = sp_data
        _x, c, w, z = take_batch(loader)
        spo = SPOPlus(optmodel, processes=1, solve_ratio=0.5, dataset=dataset)
        cp = (c * 1.2).clone().detach().requires_grad_(True)
        loss = spo(cp, c, w, z)
        loss.backward()
        assert spo.solpool.shape[0] >= 1
        assert torch.isfinite(cp.grad).all()


# ============================================================
# 3. Deep correctness gates
# ============================================================


@requires_gurobi
class TestSPOPlusGradient:
    """SPO+ autograd subgradient against its closed form 2·(w_true - w_spo)/n."""

    def _setup(self, seed=0):
        from pyepo.func.surrogate import SPOPlus
        from pyepo.model.grb.shortestpath import shortestPathModel

        rng = np.random.RandomState(seed)
        model = shortestPathModel(grid=(3, 3))
        n, d = 4, model.num_cost
        c_true = torch.as_tensor(rng.rand(n, d) + 0.1, dtype=torch.float32)
        w_true = np.zeros((n, d), dtype=np.float32)
        z_true = np.zeros((n, 1), dtype=np.float32)
        for i in range(n):
            model.setObj(c_true[i].numpy())
            sol, obj = model.solve()
            w_true[i] = np.asarray(sol, dtype=np.float32)
            z_true[i, 0] = obj
        return (
            SPOPlus(model, processes=1),
            model,
            c_true,
            torch.from_numpy(w_true),
            torch.from_numpy(z_true),
        )

    def test_subgradient_matches_closed_form(self):
        spo, model, c_true, w_true, z_true = self._setup(0)
        cp = (c_true * 1.3).clone().detach().requires_grad_(True)
        spo(cp, c_true, w_true, z_true).backward()
        spo_costs = 2 * cp.detach() - c_true
        w_spo = np.zeros_like(cp.detach().numpy())
        for i in range(cp.shape[0]):
            model.setObj(spo_costs[i].numpy())
            sol, _ = model.solve()
            w_spo[i] = np.asarray(sol, dtype=np.float32)
        expected = 2.0 * (w_true.numpy() - w_spo) / cp.shape[0]
        np.testing.assert_allclose(cp.grad.numpy(), expected, atol=1e-4)

    def test_gradient_step_decreases_loss(self):
        spo, _, c_true, w_true, z_true = self._setup(1)
        cp = (c_true * 1.5).clone().detach().requires_grad_(True)
        loss0 = spo(cp, c_true, w_true, z_true)
        loss0.backward()
        with torch.no_grad():
            cp_new = cp - 0.1 * cp.grad
        loss1 = spo(cp_new, c_true, w_true, z_true)
        assert loss1.item() <= loss0.item() + 1e-6


def _fw_knapsack():
    """Knapsack shared by the Frank-Wolfe tests: weights [3,4,2,5], capacity 7,
    so the IP optimum picks items {0, 1}."""
    from pyepo.model.grb.knapsack import knapsackModel

    return knapsackModel(weights=[[3.0, 4.0, 2.0, 5.0]], capacity=[7.0])


@requires_gurobi
class TestRegularizedFrankWolfe:
    def test_lambd_must_be_positive(self):
        from pyepo.func.regularized import RFWO

        for bad in (0.0, -1.0):
            with pytest.raises(ValueError):
                RFWO(_fw_knapsack(), lambd=bad)

    def test_compute_regularization_includes_lambd(self):
        from pyepo.func.regularized import RFWO

        m = RFWO(_fw_knapsack(), lambd=3.0, max_iter=5)
        # Omega(y) = (lambd/2)||y||^2 = 1.5 * 0.5 = 0.75
        y = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        assert torch.allclose(m.compute_regularization(y), torch.tensor([0.75]))

    def test_all_instances_solved_each_step(self, monkeypatch):
        # away-step FW solves every instance each step: its FW gap is non-monotone,
        # so a per-instance skip would stall convergence -- no batch is sliced down
        import pyepo.func.regularized as regularized
        from pyepo.func.regularized import RFWO

        batch_sizes = []

        def fake_solve_or_cache(cp, module):
            batch_sizes.append(cp.shape[0])
            sol = torch.zeros_like(cp) if len(batch_sizes) == 1 else cp.clone()
            return sol, torch.zeros(cp.shape[0], dtype=cp.dtype, device=cp.device)

        monkeypatch.setattr(regularized, "_solve_or_cache", fake_solve_or_cache)
        m = RFWO(_fw_knapsack(), max_iter=3, tol=1e-6)
        theta = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        m._frankWolfe(theta)
        assert batch_sizes and all(bs == 2 for bs in batch_sizes)

    def test_extreme_theta_collapses_to_vertex(self):
        from pyepo.func.regularized import RFWO

        m = RFWO(_fw_knapsack(), max_iter=10, tol=1e-6)
        mu, _vertices, weights = m._frankWolfe(torch.tensor([[100.0, -100.0, -100.0, -100.0]]))
        assert mu.shape == (1, 4)
        assert mu[0, 0].item() > 0.99
        assert weights.sum(dim=1)[0].item() == pytest.approx(1.0, abs=1e-5)

    def test_mu_equals_weighted_sum_of_vertices(self):
        from pyepo.func.regularized import RFWO

        m = RFWO(_fw_knapsack(), max_iter=12, tol=1e-6)
        theta = torch.tensor([[1.0, 1.5, 2.0, 0.5], [2.0, 1.0, 1.0, 2.0]])
        mu, V, w = m._frankWolfe(theta)
        assert torch.allclose(mu, (w.unsqueeze(-1) * V).sum(dim=1), atol=1e-5)

    def test_smaller_lambd_closer_to_vertex(self):
        from pyepo.func.regularized import RFWO

        opt = _fw_knapsack()
        cp = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        ip_sol = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        sharp = RFWO(opt, lambd=0.05, max_iter=30)(cp)
        smooth = RFWO(opt, lambd=10.0, max_iter=30)(cp)
        assert (sharp - ip_sol).norm().item() < (smooth - ip_sol).norm().item()

    @pytest.mark.parametrize("lambd", [0.5, 1.0, 5.0])
    def test_backward_matches_finite_difference(self, lambd):
        from pyepo.func.regularized import RFWO

        m = RFWO(_fw_knapsack(), lambd=lambd, max_iter=30, tol=1e-8)
        torch.manual_seed(0)
        cp = (torch.rand(1, 4) * 2 + 1.0).requires_grad_(True)
        target = torch.randn_like(cp)
        (target * m(cp)).sum().backward()
        analytic = cp.grad.clone()
        eps = 5e-3
        fd = torch.zeros_like(cp)
        for j in range(cp.shape[1]):
            cp_p, cp_m = cp.detach().clone(), cp.detach().clone()
            cp_p[0, j] += eps
            cp_m[0, j] -= eps
            fd[0, j] = ((target * m(cp_p)).sum() - (target * m(cp_m)).sum()) / (2 * eps)
        np.testing.assert_allclose(
            analytic.numpy(), fd.numpy(), atol=5e-2, err_msg=f"lambd={lambd}"
        )


@requires_gurobi
class TestAwayStepFrankWolfe:
    def _sp_module(self, max_iter=10000, tol=1e-7, lambd=1.0):
        from pyepo.func.regularized import RFWO
        from pyepo.model.grb.shortestpath import shortestPathModel

        return RFWO(
            shortestPathModel(grid=(5, 5)), lambd=lambd, max_iter=max_iter, tol=tol
        )

    def test_converges_and_active_set_exact(self):
        import pyepo
        from pyepo.func import regularized

        m = self._sp_module()
        _, c = pyepo.data.shortestpath.genData(5, 5, (5, 5), seed=42)
        theta = (-1.0 / m.lambd) * torch.tensor(np.asarray(c, np.float64) * 1.3)
        mu, V, W = m._frankWolfe(theta)
        # mu is exactly the weighted active set and the weights form a simplex
        recon = torch.einsum("bc,bcv->bv", W, V)
        assert float((recon - mu).norm(dim=-1).max()) < 1e-10
        assert torch.allclose(W.sum(-1), torch.ones(W.shape[0], dtype=W.dtype), atol=1e-9)
        # converged far past vanilla's ~2e-3 stall (a fresh-LMO gap re-measure is
        # tie-break-noisy at the 1e-7 level, so check well above tol)
        v_fw, _ = regularized._solve_or_cache(-1.0 * (theta - mu), m)
        gap = ((mu - theta) * (mu - v_fw.to(mu))).sum(-1)
        assert float(gap.max()) < 1e-5

    def test_gradient_matches_true_face(self):
        import pyepo
        from pyepo.func.regularized import RFWO
        from pyepo.model.grb.shortestpath import shortestPathModel

        model = shortestPathModel(grid=(5, 5))
        _, c = pyepo.data.shortestpath.genData(5, 5, (5, 5), seed=42)
        pred = torch.tensor(np.asarray(c, np.float64) * 1.3)
        target = torch.tensor(np.random.RandomState(3).randn(*pred.shape))

        def grad_at(max_iter):
            m = RFWO(model, lambd=1.0, max_iter=max_iter, tol=1e-9)
            p = pred.clone().requires_grad_(True)
            (m(p) * target).sum().backward()
            return p.grad, m

        g_ref, _ = grad_at(10000)
        g, m = grad_at(2000)
        _, _, w = m._frankWolfe(-1.0 * pred)
        # gradient is stable to the converged reference (vanilla floors ~3% off)
        assert float((g - g_ref).norm() / g_ref.norm()) < 1e-4
        # the active set is the true support, not a bloated one
        assert int((w > 0).sum(-1).max()) <= pred.shape[1] + 1

    def test_no_overflow_and_early_exit(self):
        import pyepo
        from pyepo.func import regularized

        m = self._sp_module(max_iter=10000, tol=1e-7)
        _, c = pyepo.data.shortestpath.genData(8, 5, (5, 5), seed=7)
        theta = (-1.0 / m.lambd) * torch.tensor(np.asarray(c, np.float64) * 1.3)
        mu, V, W = m._frankWolfe(theta)
        # active set never spilled past the bounded buffer
        assert int((W > 0).sum(-1).max()) < V.shape[1]
        # converged well within the cap (early-exit fired, did not run 10000 iters)
        v_fw, _ = regularized._solve_or_cache(-1.0 * (theta - mu), m)
        assert float(((mu - theta) * (mu - v_fw.to(mu))).sum(-1).max()) < 1e-5


@requires_gurobi
class TestRegularizedFrankWolfeFenchelYoung:
    def test_forward_matches_formula(self):
        from pyepo.func.regularized import RFY

        lambd = 1.7
        fy = RFY(
            _fw_knapsack(), lambd=lambd, max_iter=30, tol=1e-8, reduction="none"
        )
        cp = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
        w = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        loss = fy(cp, w)
        r_sol = fy._frankWolfe(cp / lambd)
        omega_w = 0.5 * lambd * (w**2).sum(dim=-1)
        omega_r = 0.5 * lambd * (r_sol**2).sum(dim=-1)
        expected = omega_w - omega_r + torch.sum(cp * (r_sol - w), dim=1)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_backward_matches_subgradient(self):
        from pyepo.func.regularized import RFY

        lambd = 1.0
        fy = RFY(
            _fw_knapsack(), lambd=lambd, max_iter=30, tol=1e-8, reduction="mean"
        )
        cp = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        w = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        fy(cp, w).backward()
        with torch.no_grad():
            expected = (fy._frankWolfe(cp.detach() / lambd) - w) / cp.shape[0]
        assert torch.allclose(cp.grad, expected, atol=1e-5)


# ------------------------------------------------------------
# Independent ground-truth gates for the remaining losses
# ------------------------------------------------------------
# Autograd gradient checked against an independent reference, per loss kind:
#   - value-differentiable losses -> finite difference of the loss value
#   - estimator-gradient ops       -> a separate re-solve of the estimator


class TestGradientTruthFD:
    """Autograd gradient == finite difference of the loss value."""

    @pytest.mark.parametrize("name", list(FD_TRUTH))
    def test_grad_matches_finite_difference(self, name, sp_truth):
        optmodel, dataset, loader = sp_truth
        _kind, build, sig = LOSS_REGISTRY[name]
        flags = FD_TRUTH[name]
        _x, c, w, z = take_batch(loader)
        mod = build(optmodel, dataset, "mean")
        # freeze the pool
        if "freeze_pool" in flags:
            mod.solve_ratio = 0.0

        def value(cp_np):
            with torch.no_grad():
                return float(call_op(mod, sig, torch.tensor(cp_np, dtype=torch.float32), c, w, z))

        cp0 = (c * 1.2).numpy()
        cp = torch.tensor(cp0, dtype=torch.float32, requires_grad=True)
        call_op(mod, sig, cp, c, w, z).backward()
        np.testing.assert_allclose(cp.grad.numpy(), finite_diff_grad(value, cp0), atol=3e-2)


def _gaussian_noise(mod, cp):
    """Additive Gaussian noise for a perturbed* op (seeded from mod.seed)."""
    gen = torch.Generator()
    gen.manual_seed(mod.seed)
    return torch.randn(
        (cp.shape[0], mod.n_samples, cp.shape[1]), dtype=torch.float32, generator=gen
    )


def _solve_3d_batch(optmodel, ptb_c):
    """Solve perturbed 3D costs (batch, n_samples, vars) via the shared solver."""
    from pyepo.func.utils import _solve_batch

    b, n, d = ptb_c.shape
    sols, _ = _solve_batch(ptb_c.reshape(-1, d), optmodel, 1, None)
    return sols.reshape(b, n, d)


def _perturb_torch(cp, noises, sigma, mul):
    """Additive or multiplicative log-normal perturbation of clean cost (b, d)."""
    if mul:
        return cp.unsqueeze(1) * torch.exp(sigma * noises - 0.5 * sigma**2)
    return cp.unsqueeze(1) + sigma * noises


def _grad_scale_torch(cp, n, sigma, mul):
    """Divisor of the perturbed reward estimator: n*sigma (additive) or n*denom_safe (multiplicative)."""
    from pyepo.utils import _EPS

    if not mul:
        return n * sigma + _EPS
    denom = sigma * cp
    denom_safe = torch.where(
        denom.abs() < _EPS,
        torch.where(denom >= 0, torch.full_like(denom, _EPS), torch.full_like(denom, -_EPS)),
        denom,
    )
    return n * denom_safe


class TestSolutionGradientTruth:
    """Estimator solution-ops: autograd gradient == an independent re-solve of the estimator."""

    def _setup(self, sp_truth, name):
        optmodel, dataset, loader = sp_truth
        _kind, build, _sig = LOSS_REGISTRY[name]
        _x, c, _w, _z = take_batch(loader)
        mod = build(optmodel, dataset, None)
        cp = (c * 1.2).clone().detach()
        torch.manual_seed(0)
        target = torch.randn_like(cp)
        return optmodel, mod, cp, target

    def test_negative_identity(self, sp_truth):
        _om, mod, cp, target = self._setup(sp_truth, "NID")
        cpg = cp.clone().requires_grad_(True)
        (mod(cpg) * target).sum().backward()
        # MINIMIZE: signed-identity Jacobian
        assert torch.allclose(cpg.grad, -target, atol=1e-5)

    def test_blackbox_opt(self, sp_truth):
        from pyepo.func.utils import _solve_batch

        optmodel, mod, cp, target = self._setup(sp_truth, "DBB")
        cpg = cp.clone().requires_grad_(True)
        (mod(cpg) * target).sum().backward()
        # interpolation gradient (w*(cp + lambd*d) - w*(cp)) / lambd
        sol_p, _ = _solve_batch(cp, optmodel, 1, None)
        sol_q, _ = _solve_batch(cp + mod.lambd * target, optmodel, 1, None)
        expected = (sol_q - sol_p) / mod.lambd
        assert torch.allclose(cpg.grad, expected, atol=1e-3)

    @pytest.mark.parametrize("mul", [False, True])
    def test_perturbed_opt(self, sp_truth, mul):
        optmodel, mod, cp, target = self._setup(sp_truth, "DPOMul" if mul else "DPO")
        n, sigma = mod.n_samples, mod.sigma
        cpg = cp.clone().requires_grad_(True)
        (mod(cpg) * target).sum().backward()
        noises = _gaussian_noise(mod, cp)
        ptb_sols = _solve_3d_batch(optmodel, _perturb_torch(cp, noises, sigma, mul))
        reward = torch.einsum("bnd,bd->bn", ptb_sols, target)
        if mod.variance_reduction and n > 1:  # leave-one-out baseline
            reward = (reward - reward.mean(dim=1, keepdim=True)) * (n / (n - 1))
        expected = torch.einsum("bnd,bn->bd", noises, reward) / _grad_scale_torch(cp, n, sigma, mul)
        assert torch.allclose(cpg.grad, expected, atol=1e-3)

    def test_implicit_mle(self, sp_truth):
        from pyepo.func.utils import sumGammaDistribution
        from pyepo.utils import _EPS

        optmodel, mod, cp, target = self._setup(sp_truth, "IMLE")
        n, sigma, lambd = mod.n_samples, mod.sigma, mod.lambd
        cpg = cp.clone().requires_grad_(True)
        (mod(cpg) * target).sum().backward()
        # reproduce Sum-of-Gamma noise (fresh distribution, same default seed)
        noises = sumGammaDistribution(kappa=5).sample(
            size=(cp.shape[0], n, cp.shape[1]), device=torch.device("cpu"), dtype=torch.float32
        )
        ptb_c = cp.unsqueeze(1) + sigma * noises
        ptb_sols = _solve_3d_batch(optmodel, ptb_c)
        ptb_sols_pos = _solve_3d_batch(optmodel, ptb_c + lambd * target.unsqueeze(1))
        expected = (ptb_sols_pos - ptb_sols).mean(dim=1) / (lambd + _EPS)
        assert torch.allclose(cpg.grad, expected, atol=1e-3)

    def test_adaptive_implicit_mle(self, sp_truth):
        from pyepo.func.utils import sumGammaDistribution
        from pyepo.utils import _EPS

        optmodel, mod, cp, target = self._setup(sp_truth, "AIMLE")
        n, sigma = mod.n_samples, mod.sigma
        cpg = cp.clone().requires_grad_(True)
        (mod(cpg) * target).sum().backward()
        noises = sumGammaDistribution(kappa=5).sample(
            size=(cp.shape[0], n, cp.shape[1]), device=torch.device("cpu"), dtype=torch.float32
        )
        ptb_c = cp.unsqueeze(1) + sigma * noises
        ptb_sols = _solve_3d_batch(optmodel, ptb_c)
        # adaptive lambda (alpha = 1.0)
        lambd = cp.norm() / target.norm()
        ptb_sols_pos = _solve_3d_batch(optmodel, ptb_c + lambd * target.unsqueeze(1))
        expected = (ptb_sols_pos - ptb_sols).mean(dim=1) / (lambd + _EPS)
        assert torch.allclose(cpg.grad, expected, atol=1e-3)


class TestLossGradientTruth:
    """Solve-in-forward losses: autograd gradient == an independent re-solve of the subgradient."""

    def _setup(self, sp_truth, name):
        optmodel, dataset, loader = sp_truth
        _kind, build, _sig = LOSS_REGISTRY[name]
        _x, c, w, _z = take_batch(loader)
        mod = build(optmodel, dataset, "mean")
        cp = (c * 1.2).clone().detach()
        return optmodel, mod, cp, c, w

    def test_perturbation_gradient(self, sp_truth):
        from pyepo.func.utils import _solve_batch
        from pyepo.utils import _EPS

        optmodel, mod, cp, c, _w = self._setup(sp_truth, "PG")
        b = cp.shape[0]
        cpg = cp.clone().requires_grad_(True)
        mod(cpg, c).backward()
        # backward differencing: (w*(cp) - w*(cp - sigma*c)) / sigma, mean -> /b
        w_sol, _ = _solve_batch(cp, optmodel, 1, None)
        wm_sol, _ = _solve_batch(cp - mod.sigma * c, optmodel, 1, None)
        expected = (w_sol - wm_sol) / (mod.sigma + _EPS) / b  # sign = +1 (MINIMIZE)
        assert torch.allclose(cpg.grad, expected, atol=1e-3)

    @pytest.mark.parametrize("mul", [False, True])
    def test_perturbed_fenchel_young(self, sp_truth, mul):
        optmodel, mod, cp, _c, w = self._setup(sp_truth, "PFYMul" if mul else "PFY")
        b = cp.shape[0]
        cpg = cp.clone().requires_grad_(True)
        mod(cpg, w).backward()
        noises = _gaussian_noise(mod, cp)
        ptb_sols = _solve_3d_batch(optmodel, _perturb_torch(cp, noises, mod.sigma, mul))
        if mul:
            factor = torch.exp(mod.sigma * noises - 0.5 * mod.sigma**2)
            e_sol = (ptb_sols * factor).mean(dim=1)
        else:
            e_sol = ptb_sols.mean(dim=1)
        expected = (w - e_sol) / b  # MINIMIZE residual w - E[sol]
        assert torch.allclose(cpg.grad, expected, atol=1e-3)


# ============================================================
# CaVE (cone-aligned cosine) — needs binding constraints + Clarabel
# ============================================================


@requires_gurobi
@requires_clarabel
class TestCaVE:
    @pytest.fixture
    def setup(self):
        from pyepo.func.cave import CaVE
        from pyepo.model.grb.shortestpath import shortestPathModel

        model = shortestPathModel(grid=(2, 3))  # 7 edges
        d = model.num_cost
        torch.manual_seed(0)
        pred_cost = torch.randn(2, d, requires_grad=True)
        tight_ctrs = torch.randn(2, 3, d)
        return CaVE, model, pred_cost, tight_ctrs

    def test_default_scalar_in_range(self, setup):
        cave, model, pred, ctrs = setup
        loss = cave(model, processes=1, reduction="mean")(pred, ctrs)
        assert loss.dim() == 0
        assert -1e-6 <= loss.item() <= 2.0 + 1e-6  # cosine distance in [0, 2]

    def test_reduction_none_and_sum(self, setup):
        cave, model, pred, ctrs = setup
        none = cave(model, processes=1, reduction="none")(pred, ctrs)
        assert none.shape == (2,)
        total = cave(model, processes=1, reduction="sum")(pred, ctrs)
        mean = cave(model, processes=1, reduction="mean")(pred, ctrs)
        np.testing.assert_allclose(total.item(), mean.item() * 2, atol=1e-5)

    def test_gradient_flows(self, setup):
        cave, model, pred, ctrs = setup
        cave(model, processes=1, reduction="mean")(pred, ctrs).backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape
        assert torch.isfinite(pred.grad).all()

    def test_grad_matches_fixed_proj_fd(self, setup):
        # finite difference of 1 - cos(sign*pred, proj) with proj held fixed
        from torch.nn import functional as F

        from pyepo.func.cave import _clarabel_project_batch

        cave_cls, model, pred, ctrs = setup
        cave = cave_cls(model, processes=1, reduction="mean")
        sign = cave._sign
        cp0 = pred.detach().numpy().astype(np.float32)
        # autograd gradient
        cp = torch.tensor(cp0, requires_grad=True)
        cave(cp, ctrs).backward()
        # reproduce the (fixed) projection at the un-perturbed pred
        proj = _clarabel_project_batch(
            sign * torch.tensor(cp0), ctrs.detach().cpu(), max_iter=cave.max_iter
        )

        def value(p_np):
            s = sign * torch.tensor(p_np, dtype=torch.float32)
            return float((1.0 - F.cosine_similarity(s, proj, dim=1)).mean())

        np.testing.assert_allclose(cp.grad.numpy(), finite_diff_grad(value, cp0), atol=3e-2)


# ============================================================
# perturbed internals (pure tensor math, no solver)
# ============================================================


class TestPerturbedInternals:
    def test_variance_reduction_leave_one_out(self):
        from pyepo.func.perturbed import DPO

        ptb = DPO.__new__(DPO)
        ptb.variance_reduction = True
        reward = torch.tensor([[1.0, 2.0, 4.0], [3.0, 3.0, 9.0]])
        n = reward.shape[1]
        expected = n * (reward - reward.mean(dim=1, keepdim=True)) / (n - 1)
        assert torch.allclose(ptb._apply_variance_reduction(reward), expected)

    def test_variance_reduction_single_sample_noop(self):
        from pyepo.func.perturbed import DPO

        ptb = DPO.__new__(DPO)
        ptb.variance_reduction = True
        reward = torch.tensor([[1.0], [3.0]])
        assert torch.equal(ptb._apply_variance_reduction(reward), reward)

    def test_mul_uses_weighted_expected_solution(self):
        from pyepo.func.perturbed import PFYMul

        pfy = PFYMul.__new__(PFYMul)
        pfy.sigma = 0.5
        noises = torch.tensor([[[0.0, 1.0, -1.0], [0.5, -0.5, 0.25]]])
        ptb_sols = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]])
        factor = torch.exp(pfy.sigma * noises - 0.5 * pfy.sigma**2)
        expected = (ptb_sols * factor).mean(dim=1)
        assert torch.allclose(
            pfy._calculate_expected_solution(None, None, ptb_sols, noises), expected
        )
