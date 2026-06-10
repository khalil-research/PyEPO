#!/usr/bin/env python
"""Tests for pyepo.func autograd losses, helpers, and the optModule base.

Grouped neutral -> both -> torch:

1. neutral: pure / mock helpers (solution pool, cache selection, sum-of-gamma
   noise, solution check, perturbed estimator internals) — no solver.
2. both: the forward/backward contract applied to every loss on both frontends
   via `contract_backend` over LOSS_REGISTRY / JAX_LOSS_REGISTRY. Solution ops
   return (batch, vars) with finite gradients; loss ops return a scalar that
   backpropagates and honours reduction = mean/sum/none.
3. torch: init validation, MAXIMIZE/caching behaviour, and deep correctness
   gates comparing an autograd gradient to its closed form or a finite-
   difference Jacobian (SPO+, regularized Frank-Wolfe, the estimator ops, CaVE).

The jax-only correctness gates live in test_55_jax.py.
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
    FD_LOSSES,
    LOSS_OPS,
    LOSS_REGISTRY,
    SOLUTION_OPS,
    call_op,
    finite_diff_grad,
    requires_clarabel,
    requires_gurobi,
    requires_jax,
    solver_atol,
    take_batch,
)

# ============================================================
# neutral: pure helpers (no solver)
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


class TestPerturbedInternals:
    """Perturbed estimator internals (pure tensor math, no solver)."""

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


# ============================================================
# both: forward/backward contract for every loss (torch & jax frontends)
# ============================================================
# `contract_backend` selects the frontend (LOSS_REGISTRY / JAX_LOSS_REGISTRY);
# the assertions are identical. Solution-returning ops yield (batch, vars);
# loss-returning ops yield a scalar and honour reduction = mean/sum/none.


@requires_gurobi
class TestForwardBackwardContract:
    """Solution ops return (batch, vars) with finite grad; loss ops return a
    scalar that backpropagates and honours the reduction modes."""

    @pytest.mark.parametrize("name", SOLUTION_OPS)
    def test_solution_forward_and_backward(self, name, contract_backend, sp_data):
        be = contract_backend
        optmodel, dataset, loader = sp_data
        _kind, build, sig = be.registry[name]
        _x, c, w, z = take_batch(loader)
        cp, c2, w2, z2 = be.inputs(c, w, z)
        op = build(optmodel, dataset, "mean")
        out = be.forward(op, sig, cp, c2, w2, z2)
        assert be.shape(out) == be.shape(cp)
        assert be.finite(out)
        g = be.grad(op, sig, cp, c2, w2, z2)
        assert be.shape(g) == be.shape(cp)
        assert be.finite(g)

    @pytest.mark.parametrize("name", LOSS_OPS)
    def test_loss_scalar_and_backward(self, name, contract_backend, sp_data):
        be = contract_backend
        optmodel, dataset, loader = sp_data
        _kind, build, sig = be.registry[name]
        _x, c, w, z = take_batch(loader)
        cp, c2, w2, z2 = be.inputs(c, w, z)
        op = build(optmodel, dataset, "mean")
        out = be.forward(op, sig, cp, c2, w2, z2)
        assert be.ndim(out) == 0
        g = be.grad(op, sig, cp, c2, w2, z2)
        assert be.finite(g)

    @pytest.mark.parametrize("name", LOSS_OPS)
    def test_loss_reduction_modes(self, name, contract_backend, sp_data):
        be = contract_backend
        optmodel, dataset, loader = sp_data
        _kind, build, sig = be.registry[name]
        _x, c, w, z = take_batch(loader)
        cp, c2, w2, z2 = be.inputs(c, w, z)
        # most losses reduce to (batch,); lsLTR keeps a (batch, pool) grid
        none = be.forward(build(optmodel, dataset, "none"), sig, cp, c2, w2, z2)
        assert be.shape(none)[0] == be.shape(cp)[0]
        mean = be.forward(build(optmodel, dataset, "mean"), sig, cp, c2, w2, z2)
        total = be.forward(build(optmodel, dataset, "sum"), sig, cp, c2, w2, z2)
        np.testing.assert_allclose(float(be.to_np(mean)), float(be.to_np(none).mean()), atol=1e-5)
        np.testing.assert_allclose(float(be.to_np(total)), float(be.to_np(none).sum()), atol=1e-5)


# ============================================================
# both: optModule init validation (torch & jax frontends)
# ============================================================


@pytest.fixture(params=[pytest.param("torch"), pytest.param("jax", marks=requires_jax)])
def func_frontend(request):
    """The pyepo.func / pyepo.func.jax module; init validation is identical."""
    if request.param == "torch":
        import pyepo.func as mod
    else:
        import pyepo.func.jax as mod
    return mod


@requires_gurobi
class TestOptModuleInit:
    def _model(self):
        from pyepo.model.grb.shortestpath import shortestPathModel

        return shortestPathModel(grid=(3, 3))

    def test_invalid_model_type_raises(self, func_frontend):
        with pytest.raises(TypeError):
            func_frontend.SPOPlus("not_a_model")

    def test_invalid_processes_raises(self, func_frontend):
        with pytest.raises(ValueError):
            func_frontend.SPOPlus(self._model(), processes=-1)

    @pytest.mark.parametrize("ratio", [1.5, -0.1])
    def test_invalid_solve_ratio_raises(self, func_frontend, ratio):
        with pytest.raises(ValueError):
            func_frontend.SPOPlus(self._model(), solve_ratio=ratio)

    def test_solve_ratio_lt1_requires_dataset(self, func_frontend):
        with pytest.raises(TypeError):
            func_frontend.SPOPlus(self._model(), solve_ratio=0.5, dataset=None)


@requires_gurobi
class TestConstructorGuards:
    """Constructor validation shared across losses: lambda must be positive."""

    @pytest.mark.parametrize(
        "name",
        [
            "blackboxOpt",
            "implicitMLE",
            "regularizedFrankWolfeOpt",
            "regularizedFrankWolfeFenchelYoung",
        ],
    )
    def test_rejects_nonpositive_lambda(self, func_frontend, name):
        from pyepo.model.grb.shortestpath import shortestPathModel

        model = shortestPathModel(grid=(3, 3))
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError):
                getattr(func_frontend, name)(model, processes=1, lambd=bad)


# ============================================================
# torch: MAXIMIZE sense and solve-ratio caching
# ============================================================


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
# torch: deep correctness gates
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


def _fw_knapsack():
    """Knapsack shared by the Frank-Wolfe tests: weights [3,4,2,5], capacity 7,
    so the IP optimum picks items {0, 1}."""
    from pyepo.model.grb.knapsack import knapsackModel

    return knapsackModel(weights=[[3.0, 4.0, 2.0, 5.0]], capacity=[7.0])


@requires_gurobi
class TestRegularizedFrankWolfe:
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

    @pytest.mark.parametrize("name", FD_LOSSES)
    def test_grad_matches_finite_difference(self, name, sp_truth):
        optmodel, dataset, loader = sp_truth
        _kind, build, sig = LOSS_REGISTRY[name]
        _x, c, w, z = take_batch(loader)
        mod = build(optmodel, dataset, "mean")
        mod.solve_ratio = 0.0  # freeze the pool: keeps the FD value deterministic

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
    """Estimator solution-ops: autograd gradient == the estimator's closed form,
    reconstructed from an independent re-solve (or, for DPO, from the module's
    recorded solutions, since a first-order GPU solve is not reproducible)."""

    def _setup(self, sp_truth, name):
        optmodel, dataset, loader = sp_truth
        _kind, build, _sig = LOSS_REGISTRY[name]
        _x, c, _w, _z = take_batch(loader)
        mod = build(optmodel, dataset, None)
        cp = (c * 1.2).clone().detach()
        torch.manual_seed(0)
        target = torch.randn_like(cp)
        self.atol = solver_atol(optmodel)
        return optmodel, mod, cp, target

    @staticmethod
    def _imle_noise_ptb(cp, mod):
        """Perturbed costs sharing the module's Sum-of-Gamma noise (fresh draw, same default seed)."""
        from pyepo.func.utils import sumGammaDistribution

        noises = sumGammaDistribution(kappa=5).sample(
            size=(cp.shape[0], mod.n_samples, cp.shape[1]),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        return cp.unsqueeze(1) + mod.sigma * noises

    @staticmethod
    def _resolve_two_sides(optmodel, ptb_c, target, lambd):
        """Central-difference re-solve estimator: (w(+) - w(-)) / (2*lambd)."""
        from pyepo.utils import _EPS

        delta = lambd * target.unsqueeze(1)
        pos = _solve_3d_batch(optmodel, ptb_c + delta)
        neg = _solve_3d_batch(optmodel, ptb_c - delta)
        return (pos - neg).mean(dim=1) / (2 * lambd + _EPS)

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
        assert torch.allclose(cpg.grad, expected, atol=self.atol)

    @pytest.mark.parametrize("mul", [False, True])
    def test_perturbed_opt(self, sp_truth, mul, monkeypatch):
        # this gate records the solutions the module actually used instead of
        # re-solving: MPAX's GPU PDHG is not run-to-run reproducible on near-tie
        # vertices (~3e-2 between two solves of identical costs), so the
        # independently reconstructed perturbed costs are asserted against the
        # recorded solver input, and the estimator math is then checked exactly
        import pyepo.func.perturbed as perturbed

        _optmodel, mod, cp, target = self._setup(sp_truth, "DPOMul" if mul else "DPO")
        n, sigma = mod.n_samples, mod.sigma
        recorded = []
        real_solve = perturbed._solve_or_cache_3d

        def recording(ptb_c, module):
            ptb_sols = real_solve(ptb_c, module)
            recorded.append((ptb_c.detach().clone(), ptb_sols.detach().clone()))
            return ptb_sols

        monkeypatch.setattr(perturbed, "_solve_or_cache_3d", recording)
        cpg = cp.clone().requires_grad_(True)
        (mod(cpg) * target).sum().backward()
        noises = _gaussian_noise(mod, cp)
        rec_c, ptb_sols = recorded[0]
        # the module solved exactly the perturbed costs the formula prescribes
        assert torch.allclose(rec_c, _perturb_torch(cp, noises, sigma, mul), atol=1e-6)
        reward = torch.einsum("bnd,bd->bn", ptb_sols, target)
        if mod.variance_reduction and n > 1:  # leave-one-out baseline
            reward = (reward - reward.mean(dim=1, keepdim=True)) * (n / (n - 1))
        expected = torch.einsum("bnd,bn->bd", noises, reward) / _grad_scale_torch(cp, n, sigma, mul)
        assert torch.allclose(cpg.grad, expected, atol=1e-4)

    def test_implicit_mle(self, sp_truth):
        from pyepo.utils import _EPS

        optmodel, mod, cp, target = self._setup(sp_truth, "IMLE")
        cpg = cp.clone().requires_grad_(True)
        (mod(cpg) * target).sum().backward()
        ptb_c = self._imle_noise_ptb(cp, mod)
        ptb_sols = _solve_3d_batch(optmodel, ptb_c)
        ptb_sols_pos = _solve_3d_batch(optmodel, ptb_c + mod.lambd * target.unsqueeze(1))
        expected = (ptb_sols_pos - ptb_sols).mean(dim=1) / (mod.lambd + _EPS)
        assert torch.allclose(cpg.grad, expected, atol=self.atol)

    def test_adaptive_implicit_mle(self, sp_truth):
        from pyepo.utils import _EPS

        optmodel, mod, cp, target = self._setup(sp_truth, "AIMLE")
        a0 = mod.alpha
        cpg = cp.clone().requires_grad_(True)
        (mod(cpg) * target).sum().backward()
        ptb_c = self._imle_noise_ptb(cp, mod)
        ptb_sols = _solve_3d_batch(optmodel, ptb_c)
        lambd = cp.norm() / target.norm()  # adaptive lambda (alpha = 1.0)
        ptb_sols_pos = _solve_3d_batch(optmodel, ptb_c + lambd * target.unsqueeze(1))
        expected = (ptb_sols_pos - ptb_sols).mean(dim=1) / (lambd + _EPS)
        assert torch.allclose(cpg.grad, expected, atol=self.atol)
        assert mod.alpha != a0  # online alpha update fired

    def test_implicit_mle_two_sides(self, sp_truth):
        from pyepo.func.perturbed import implicitMLE

        optmodel, _mod, cp, target = self._setup(sp_truth, "IMLE")
        mod = implicitMLE(optmodel, processes=1, n_samples=3, sigma=1.0, two_sides=True)
        cpg = cp.clone().requires_grad_(True)
        (mod(cpg) * target).sum().backward()
        ptb_c = self._imle_noise_ptb(cp, mod)
        expected = self._resolve_two_sides(optmodel, ptb_c, target, mod.lambd)
        assert torch.allclose(cpg.grad, expected, atol=self.atol)

    def test_adaptive_implicit_mle_two_sides(self, sp_truth):
        from pyepo.func.perturbed import adaptiveImplicitMLE

        optmodel, _mod, cp, target = self._setup(sp_truth, "AIMLE")
        mod = adaptiveImplicitMLE(optmodel, processes=1, n_samples=3, sigma=1.0, two_sides=True)
        a0 = mod.alpha
        cpg = cp.clone().requires_grad_(True)
        (mod(cpg) * target).sum().backward()
        ptb_c = self._imle_noise_ptb(cp, mod)
        lambd = cp.norm() / target.norm()  # adaptive lambda (alpha = 1.0)
        expected = self._resolve_two_sides(optmodel, ptb_c, target, lambd)
        assert torch.allclose(cpg.grad, expected, atol=self.atol)
        assert mod.alpha != a0  # online alpha update fired


class TestLossGradientTruth:
    """Solve-in-forward losses: autograd gradient == an independent re-solve of the subgradient."""

    def _setup(self, sp_truth, name):
        optmodel, dataset, loader = sp_truth
        _kind, build, _sig = LOSS_REGISTRY[name]
        _x, c, w, _z = take_batch(loader)
        mod = build(optmodel, dataset, "mean")
        cp = (c * 1.2).clone().detach()
        self.atol = solver_atol(optmodel)
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
        assert torch.allclose(cpg.grad, expected, atol=self.atol)

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
        assert torch.allclose(cpg.grad, expected, atol=self.atol)


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
