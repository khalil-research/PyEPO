#!/usr/bin/env python
"""Tests for the JAX training frontend (pyepo.func.jax).

Correctness gate is the closed form (independent re-solve), not torch-vs-jax.
Covers both backends of batch_solve: the universal pure_callback path and the
native MPAX path. A jax-only training step confirms end-to-end gradient flow.
"""

import numpy as np
import pytest

from .conftest import (
    GRID,
    JAX_LOSS_OPS,
    JAX_LOSS_REGISTRY,
    JAX_SOLUTION_OPS,
    LOSS_REGISTRY,
    NUM_FEAT,
    call_op,
    finite_diff_grad,
    requires_clarabel,
    requires_gurobi,
    requires_mpax,
    take_batch,
)

# losses with a deterministic forward (no internal noise) -> torch-vs-jax grad parity
DETERMINISTIC_PARITY = [
    "SPOPlus",
    "PG",
    "DBB",
    "NID",
    "RFWO",
    "RFY",
    "lsLTR",
    "prLTR",
    "ptLTR",
    "NCE",
    "CMAP",
]

SEED = 42


def _sp_mpax(n):
    """mpax shortest-path model + (n, vars) float32 costs."""
    import pyepo
    from pyepo.model.mpax.shortestpath import shortestPathModel

    _x, c = pyepo.data.shortestpath.genData(n, NUM_FEAT, GRID, seed=SEED)
    return shortestPathModel(grid=GRID), np.asarray(c, np.float32)


def _sp_mpax_ds(n):
    """mpax shortest-path model + optDataset."""
    import pyepo
    from pyepo.data.dataset import optDataset
    from pyepo.model.mpax.shortestpath import shortestPathModel

    x, c = pyepo.data.shortestpath.genData(n, NUM_FEAT, GRID, seed=SEED)
    model = shortestPathModel(grid=GRID)
    return model, optDataset(model, x, c)


@requires_mpax
class TestBatchSolve:
    def test_callback_matches_native(self):
        import jax.numpy as jnp

        from pyepo.func.jax.utils import _batch_solve_callback, _batch_solve_mpax

        model, c = _sp_mpax(8)
        c_jax = jnp.asarray(c)
        sol_n, obj_n = _batch_solve_mpax(c_jax, model)
        sol_c, obj_c = _batch_solve_callback(c_jax, model)
        np.testing.assert_allclose(np.array(sol_c), np.array(sol_n), atol=1e-3)
        np.testing.assert_allclose(np.array(obj_c), np.array(obj_n), atol=1e-3)


def _spo_closed_form(model, pred, true_cost, true_sol):
    """Independent ground truth: 2*(w_true - w_spo) for MINIMIZE."""
    import jax.numpy as jnp

    from pyepo.func.jax.utils import batch_solve

    w_spo, _ = batch_solve(jnp.asarray(2.0 * pred - true_cost), model)
    return 2.0 * (true_sol - np.array(w_spo))


@requires_mpax
class TestSPOPlusJaxMpax:
    def _setup(self):
        model, ds = _sp_mpax_ds(16)
        pred = (np.asarray(ds.costs) * 1.3).astype(np.float32)
        return (
            model,
            pred,
            np.asarray(ds.costs, np.float32),
            np.asarray(ds.sols, np.float32),
            np.asarray(ds.objs, np.float32),
        )

    def test_grad_matches_closed_form(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import SPOPlus

        model, pred, tc, ts, to = self._setup()
        B = pred.shape[0]
        spo = SPOPlus(model, reduction="mean")

        def f(p):
            return spo(p, jnp.asarray(tc), jnp.asarray(ts), jnp.asarray(to))

        grad = np.array(jax.grad(f)(jnp.asarray(pred)))
        expected = _spo_closed_form(model, pred, tc, ts) / B
        np.testing.assert_allclose(grad, expected, atol=1e-3)

    def test_training_step_decreases_loss(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import SPOPlus

        model, _pred, tc, ts, to = self._setup()
        x = np.random.RandomState(1).randn(tc.shape[0], NUM_FEAT).astype(np.float32)
        rng = np.random.RandomState(2)
        params = {
            "W": jnp.asarray(0.01 * rng.randn(NUM_FEAT, tc.shape[1]).astype(np.float32)),
            "b": jnp.zeros((tc.shape[1],), jnp.float32),
        }
        xj, tcj, tsj, toj = (jnp.asarray(a) for a in (x, tc, ts, to))
        spo = SPOPlus(model, reduction="mean")

        def loss_fn(p):
            pred_cost = xj @ p["W"] + p["b"]
            return spo(pred_cost, tcj, tsj, toj)

        l0 = float(loss_fn(params))
        for _ in range(20):
            g = jax.grad(loss_fn)(params)
            params = jax.tree_util.tree_map(lambda a, d: a - 0.1 * d, params, g)
        assert float(loss_fn(params)) < l0


@requires_gurobi
class TestSPOPlusJaxCallback:
    """Non-MPAX backend: SPO+ over Gurobi via the pure_callback path."""

    def test_grad_matches_closed_form(self):
        import jax
        import jax.numpy as jnp

        import pyepo
        from pyepo.data.dataset import optDataset
        from pyepo.func.jax import SPOPlus
        from pyepo.model.grb.shortestpath import shortestPathModel

        x, c = pyepo.data.shortestpath.genData(8, NUM_FEAT, GRID, seed=SEED)
        model = shortestPathModel(grid=GRID)
        ds = optDataset(model, x, c)
        pred = (np.asarray(ds.costs) * 1.3).astype(np.float32)
        tc, ts, to = (np.asarray(a, np.float32) for a in (ds.costs, ds.sols, ds.objs))
        B = pred.shape[0]
        spo = SPOPlus(model, reduction="mean")

        def f(p):
            return spo(p, jnp.asarray(tc), jnp.asarray(ts), jnp.asarray(to))

        grad = np.array(jax.grad(f)(jnp.asarray(pred)))
        expected = _spo_closed_form(model, pred, tc, ts) / B
        np.testing.assert_allclose(grad, expected, atol=1e-4)


@requires_mpax
class TestSolveCacheHelpers:
    """Solution-pool caching helpers (pure jnp, no solver)."""

    def test_update_pool_dedups_and_appends(self):
        import jax.numpy as jnp

        from pyepo.func.jax.utils import _update_solution_pool

        pool = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        # one duplicate row, one new row
        out = _update_solution_pool(jnp.array([[1.0, 0.0], [1.0, 1.0]]), pool)
        assert int(out.shape[0]) == 3

    def test_cache_in_pass_minimize_picks_min_obj(self):
        from unittest.mock import MagicMock

        import jax.numpy as jnp

        from pyepo import EPO
        from pyepo.func.jax.utils import _cache_in_pass

        m = MagicMock()
        m.modelSense = EPO.MINIMIZE
        cost = jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        pool = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        sol, obj = _cache_in_pass(cost, m, pool)
        np.testing.assert_allclose(np.array(sol[0]), [1.0, 0.0, 0.0])
        np.testing.assert_allclose(np.array(sol[1]), [0.0, 0.0, 1.0])
        np.testing.assert_allclose(np.array(obj), [1.0, 1.0])

    def test_spoplus_caching_runs_eager(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import SPOPlus

        model, ds = _sp_mpax_ds(16)
        pred = (np.random.RandomState(0).rand(*np.asarray(ds.costs).shape) + 0.1).astype(np.float32)
        tc, ts, to = (np.asarray(a, np.float32) for a in (ds.costs, ds.sols, ds.objs))
        spo = SPOPlus(model, solve_ratio=0.5, dataset=ds)
        # force the solve-and-grow branch deterministically
        spo.solve_ratio = 1.0
        n0 = int(spo.solpool.shape[0])
        grad = np.array(
            jax.grad(lambda p: spo(p, jnp.asarray(tc), jnp.asarray(ts), jnp.asarray(to)))(
                jnp.asarray(pred)
            )
        )
        assert np.isfinite(grad).all()
        assert int(spo.solpool.shape[0]) >= n0


@requires_mpax
class TestBlackboxJax:
    def _setup(self):
        model, c = _sp_mpax(8)
        pred = (c * 1.3).astype(np.float32)
        target = np.random.RandomState(3).randn(*pred.shape).astype(np.float32)
        return model, pred, target

    def test_negative_identity_grad_is_negative_cotangent(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import negativeIdentity

        model, pred, target = self._setup()
        nid = negativeIdentity(model)
        g = jax.grad(lambda p: jnp.sum(jnp.asarray(target) * nid(p)))(jnp.asarray(pred))
        np.testing.assert_allclose(np.array(g), -target, atol=1e-6)  # MINIMIZE

    def test_blackbox_opt_grad_matches_closed_form(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import blackboxOpt
        from pyepo.func.jax.utils import batch_solve

        model, pred, target = self._setup()
        lambd = 10.0
        dbb = blackboxOpt(model, lambd=lambd)
        g = np.array(jax.grad(lambda p: jnp.sum(jnp.asarray(target) * dbb(p)))(jnp.asarray(pred)))
        # closed form: (w*(pred + lambd*d) - w*(pred)) / lambd with d = target
        wp, _ = batch_solve(jnp.asarray(pred), model)
        wq, _ = batch_solve(jnp.asarray(pred + lambd * target), model)
        expected = (np.array(wq) - np.array(wp)) / lambd
        np.testing.assert_allclose(g, expected, atol=1e-3)


@requires_mpax
class TestPerturbedJax:
    """Gate = formula re-derivation from the same seed-derived noise (not torch-vs-jax)."""

    def _model_pred(self):
        return _sp_mpax(8)

    def _gauss(self, seed, shape):
        import jax
        import jax.numpy as jnp

        _key, sub = jax.random.split(jax.random.PRNGKey(seed))
        return jax.random.normal(sub, shape, dtype=jnp.float32)

    def test_perturbed_fenchel_young_grad_matches_reference(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import perturbedFenchelYoung
        from pyepo.func.jax.utils import batch_solve

        model, c = self._model_pred()
        n_samples, sigma, seed = 5, 1.0, 0
        w_true, _ = batch_solve(jnp.asarray(c), model)
        w_true = np.array(w_true)
        pred = (c * 1.3).astype(np.float32)
        B, d = pred.shape
        noises = self._gauss(seed, (B, n_samples, d))
        ptb_c = pred[:, None, :] + sigma * noises
        sols, _ = batch_solve(jnp.asarray(ptb_c.reshape(-1, d)), model)
        e_sol = np.array(sols).reshape(B, n_samples, d).mean(1)
        expected = (w_true - e_sol) / B
        pfy = perturbedFenchelYoung(model, n_samples=n_samples, sigma=sigma, seed=seed)
        g = np.array(jax.grad(lambda p: pfy(p, jnp.asarray(w_true)))(jnp.asarray(pred)))
        np.testing.assert_allclose(g, expected, atol=1e-3)

    def test_perturbed_opt_grad_matches_reference(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import perturbedOpt
        from pyepo.func.jax.utils import batch_solve

        model, c = self._model_pred()
        n_samples, sigma, seed = 5, 1.0, 0
        pred = (c * 1.3).astype(np.float32)
        B, d = pred.shape
        target = np.random.RandomState(3).randn(B, d).astype(np.float32)
        noises = self._gauss(seed, (B, n_samples, d))
        ptb_c = pred[:, None, :] + sigma * noises
        sols, _ = batch_solve(jnp.asarray(ptb_c.reshape(-1, d)), model)
        ptb_sols = np.array(sols).reshape(B, n_samples, d)
        reward = np.einsum("bnd,bd->bn", ptb_sols, target)
        reward = (reward - reward.mean(1, keepdims=True)) * (n_samples / (n_samples - 1))
        expected = np.einsum("bnd,bn->bd", np.array(noises), reward) / (n_samples * sigma)
        po = perturbedOpt(model, n_samples=n_samples, sigma=sigma, seed=seed)
        g = np.array(jax.grad(lambda p: jnp.sum(jnp.asarray(target) * po(p)))(jnp.asarray(pred)))
        np.testing.assert_allclose(g, expected, atol=1e-3)

    def test_perturbed_opt_mul_grad_matches_reference(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import perturbedOptMul
        from pyepo.func.jax.utils import batch_solve

        model, c = self._model_pred()
        n_samples, sigma, seed = 5, 0.5, 0
        pred = (c * 1.3).astype(np.float32)
        B, d = pred.shape
        target = np.random.RandomState(3).randn(B, d).astype(np.float32)
        noises = self._gauss(seed, (B, n_samples, d))
        # build ptb_c with jnp.exp to match the impl exactly (vertex-flip gotcha)
        ptb_c = jnp.asarray(pred)[:, None, :] * jnp.exp(sigma * noises - 0.5 * sigma**2)
        sols, _ = batch_solve(ptb_c.reshape(-1, d), model)
        ptb_sols = np.array(sols).reshape(B, n_samples, d)
        reward = np.einsum("bnd,bd->bn", ptb_sols, target)
        reward = (reward - reward.mean(1, keepdims=True)) * (n_samples / (n_samples - 1))
        denom = n_samples * sigma * pred
        expected = np.einsum("bnd,bn->bd", np.array(noises), reward) / denom
        po = perturbedOptMul(model, n_samples=n_samples, sigma=sigma, seed=seed)
        g = np.array(jax.grad(lambda p: jnp.sum(jnp.asarray(target) * po(p)))(jnp.asarray(pred)))
        np.testing.assert_allclose(g, expected, atol=1e-3)

    def test_perturbed_fenchel_young_mul_grad_matches_reference(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import perturbedFenchelYoungMul
        from pyepo.func.jax.utils import batch_solve

        model, c = self._model_pred()
        n_samples, sigma, seed = 5, 0.5, 0
        w_true = np.array(batch_solve(jnp.asarray(c), model)[0])
        pred = (c * 1.3).astype(np.float32)
        B, d = pred.shape
        noises = self._gauss(seed, (B, n_samples, d))
        factor = jnp.exp(sigma * noises - 0.5 * sigma**2)
        ptb_c = jnp.asarray(pred)[:, None, :] * factor
        sols, _ = batch_solve(ptb_c.reshape(-1, d), model)
        ptb_sols = jnp.asarray(np.array(sols).reshape(B, n_samples, d))
        e_sol = np.array((ptb_sols * factor).mean(axis=1))
        expected = (w_true - e_sol) / B
        pfy = perturbedFenchelYoungMul(model, n_samples=n_samples, sigma=sigma, seed=seed)
        g = np.array(jax.grad(lambda p: pfy(p, jnp.asarray(w_true)))(jnp.asarray(pred)))
        np.testing.assert_allclose(g, expected, atol=1e-3)


@requires_mpax
class TestMaskPred:
    """Partial-prediction masking: zero perturbation on non-predicted cost positions."""

    def test_none_is_noop(self):
        from unittest.mock import MagicMock

        import jax.numpy as jnp

        from pyepo.func.jax.perturbed import _mask_pred

        m = MagicMock()
        m.c_pred_index = None
        noises = jnp.ones((1, 2, 4))
        np.testing.assert_array_equal(np.array(_mask_pred(noises, m)), np.ones((1, 2, 4)))

    def test_masks_non_predicted_positions(self):
        from unittest.mock import MagicMock

        import jax.numpy as jnp

        from pyepo.func.jax.perturbed import _mask_pred

        m = MagicMock()
        m.c_pred_index = np.array([0, 2])
        out = np.array(_mask_pred(jnp.ones((1, 2, 4)), m))
        np.testing.assert_array_equal(out[..., [0, 2]], 1.0)  # predicted positions kept
        np.testing.assert_array_equal(out[..., [1, 3]], 0.0)  # fixed positions zeroed


@requires_mpax
class TestImplicitMLEJax:
    """I-MLE / AI-MLE: gradient == an independent re-solve along the seed-derived noise."""

    def _setup(self):
        """Returns (model, pred, target, ptb_c) sharing the seed-0 Sum-of-Gamma noise."""
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax.perturbed import _sum_gamma_sample

        model, c = _sp_mpax(8)
        pred = (c * 1.3).astype(np.float32)
        b, d = pred.shape
        _key, sub = jax.random.split(jax.random.PRNGKey(0))
        noises = _sum_gamma_sample(sub, 5.0, 10, (b, 5, d))
        ptb_c = jnp.asarray(pred)[:, None, :] + 1.0 * noises
        target = np.random.RandomState(3).randn(b, d).astype(np.float32)
        return model, pred, target, ptb_c

    @staticmethod
    def _resolve_grad(module, target, ptb_c, lambd):
        import jax.numpy as jnp

        from pyepo.func.jax.perturbed import solve_or_cache_3d

        ptb_sols = np.array(solve_or_cache_3d(ptb_c, module))
        sols_pos = np.array(
            solve_or_cache_3d(ptb_c + lambd * jnp.asarray(target)[:, None, :], module)
        )
        return (sols_pos - ptb_sols).mean(axis=1) / lambd

    def test_implicit_mle_grad_matches_reference(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import implicitMLE

        model, pred, target, ptb_c = self._setup()
        lambd = 10.0
        imle = implicitMLE(model, n_samples=5, sigma=1.0, lambd=lambd, kappa=5.0, seed=0)
        expected = self._resolve_grad(imle, target, ptb_c, lambd)
        g = np.array(jax.grad(lambda p: jnp.sum(jnp.asarray(target) * imle(p)))(jnp.asarray(pred)))
        np.testing.assert_allclose(g, expected, atol=1e-3)

    def test_adaptive_grad_matches_reference_and_alpha_adapts(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import adaptiveImplicitMLE

        model, pred, target, ptb_c = self._setup()
        aimle = adaptiveImplicitMLE(model, n_samples=5, sigma=1.0, kappa=5.0, seed=0)
        a0 = aimle.alpha
        lambd = a0 * float(np.linalg.norm(pred)) / float(np.linalg.norm(target))
        expected = self._resolve_grad(aimle, target, ptb_c, lambd)
        g = np.array(jax.grad(lambda p: jnp.sum(jnp.asarray(target) * aimle(p)))(jnp.asarray(pred)))
        np.testing.assert_allclose(g, expected, atol=1e-3)
        assert aimle.alpha != a0


@requires_gurobi
class TestRegularizedJax:
    """Regularized FW over an exact Gurobi LMO; AFW cross-checked against torch."""

    def _knapsack(self):
        from pyepo.model.grb.knapsack import knapsackModel

        return knapsackModel(weights=[[3.0, 4.0, 2.0, 5.0]], capacity=[7.0])

    def test_fy_grad_matches_danskin_residual(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import regularizedFrankWolfeFenchelYoung
        from pyepo.func.jax.regularized import _frank_wolfe_active

        model = self._knapsack()
        lambd = 1.0
        cp = np.array([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]], np.float32)
        w = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]], np.float32)
        B = cp.shape[0]
        fy = regularizedFrankWolfeFenchelYoung(model, lambd=lambd, max_iter=100, tol=1e-8)
        g = np.array(jax.grad(lambda p: fy(p, jnp.asarray(w)))(jnp.asarray(cp)))
        # MAXIMIZE: theta = pred/lambd, Danskin residual diff = r_sol - w
        r_sol = np.array(_frank_wolfe_active(jnp.asarray(cp) / lambd, fy)[0])
        expected = (r_sol - w) / B
        np.testing.assert_allclose(g, expected, atol=1e-3)

    def test_caching_reads_pool_and_grows(self):
        # solve_ratio<1 seeds a vertex pool: cached passes read it, exact passes grow it
        import jax
        import jax.numpy as jnp

        from pyepo.data.dataset import optDataset
        from pyepo.func.jax import regularizedFrankWolfeOpt as JOpt

        model = self._knapsack()
        rng = np.random.RandomState(0)
        c = (rng.rand(6, model.num_cost) + 0.5).astype(np.float32)
        x = rng.rand(6, 3).astype(np.float32)
        ds = optDataset(model, x, c)
        target = rng.randn(6, model.num_cost).astype(np.float32)
        opt = JOpt(model, lambd=1.0, max_iter=50, tol=1e-8, solve_ratio=0.5, dataset=ds)
        n0 = int(opt.solpool.shape[0])

        def grad(cp):
            return np.array(
                jax.grad(lambda p: jnp.sum(jnp.asarray(target) * opt(p)))(jnp.asarray(cp))
            )

        # force a cached forward: runs on the frozen pool, no growth
        opt.solve_ratio = 0.0
        g_cache = grad(c * 1.2)
        assert np.isfinite(g_cache).all()
        assert int(opt.solpool.shape[0]) == n0
        # force an exact forward: solves the LMO and may grow the pool
        opt.solve_ratio = 1.0
        g_exact = grad(c * 1.2)
        assert np.isfinite(g_exact).all()
        assert int(opt.solpool.shape[0]) >= n0

    def test_multiprocessing_lmo_matches_single_core(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import regularizedFrankWolfeOpt as JOpt

        cp = np.array([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]], np.float32)
        target = np.random.RandomState(0).randn(*cp.shape).astype(np.float32)

        def grad_with(n_proc):
            opt = JOpt(self._knapsack(), lambd=1.0, max_iter=50, tol=1e-8, processes=n_proc)
            return np.array(
                jax.grad(lambda p: jnp.sum(jnp.asarray(target) * opt(p)))(jnp.asarray(cp))
            )

        np.testing.assert_allclose(grad_with(2), grad_with(1), atol=1e-4)

    def test_opt_forward_and_backward_match_torch(self):
        import jax
        import jax.numpy as jnp
        import torch

        from pyepo.func.jax import regularizedFrankWolfeOpt as JOpt
        from pyepo.func.regularized import regularizedFrankWolfeOpt as TOpt

        lambd = 1.0
        cp = np.array([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]], np.float32)
        target = np.random.RandomState(0).randn(*cp.shape).astype(np.float32)
        # jax (exact Gurobi LMO via callback)
        jopt = JOpt(self._knapsack(), lambd=lambd, max_iter=200, tol=1e-8)
        mu_j = np.array(jopt(jnp.asarray(cp)))
        g_j = np.array(jax.grad(lambda p: jnp.sum(jnp.asarray(target) * jopt(p)))(jnp.asarray(cp)))
        # torch reference (same exact LMO)
        topt = TOpt(self._knapsack(), lambd=lambd, max_iter=200, tol=1e-8)
        cpt = torch.tensor(cp, requires_grad=True)
        mu_t = topt(cpt)
        (torch.as_tensor(target) * mu_t).sum().backward()
        np.testing.assert_allclose(mu_j, mu_t.detach().numpy(), atol=1e-3)
        np.testing.assert_allclose(g_j, cpt.grad.numpy(), atol=1e-3)


@requires_gurobi
class TestConstructorGuards:
    """Constructor validation shared across losses."""

    @pytest.mark.parametrize(
        "name",
        [
            "blackboxOpt",
            "implicitMLE",
            "regularizedFrankWolfeOpt",
            "regularizedFrankWolfeFenchelYoung",
        ],
    )
    def test_rejects_nonpositive_lambda(self, name):
        import pyepo.func.jax as J
        from pyepo.model.grb.shortestpath import shortestPathModel

        model = shortestPathModel(grid=GRID)
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError):
                getattr(J, name)(model, lambd=bad)


@requires_mpax
class TestRankContrastiveJax:
    """Pool/contrastive losses gated by finite difference (reused from conftest)."""

    def _data(self):
        model, ds = _sp_mpax_ds(12)
        pred = (np.asarray(ds.costs) * 1.3).astype(np.float32)
        return model, ds, pred, np.asarray(ds.costs, np.float32), np.asarray(ds.sols, np.float32)

    @pytest.mark.parametrize(
        "name,arg",
        [
            ("listwiseLearningToRank", "cost"),
            ("pairwiseLearningToRank", "cost"),
            ("pointwiseLearningToRank", "cost"),
            ("noiseContrastiveEstimation", "sol"),
            ("contrastiveMAP", "sol"),
        ],
    )
    def test_grad_matches_fd(self, name, arg):
        import jax
        import jax.numpy as jnp

        import pyepo.func.jax as J

        model, ds, pred, true_cost, true_sol = self._data()
        # solve_ratio=0 freezes the pool so the finite difference is deterministic
        loss = getattr(J, name)(model, dataset=ds, solve_ratio=0)
        second = jnp.asarray(true_cost if arg == "cost" else true_sol)

        def f(p):
            return float(loss(jnp.asarray(p), second))

        g_auto = np.array(jax.grad(lambda p: loss(p, second))(jnp.asarray(pred)))
        np.testing.assert_allclose(g_auto, finite_diff_grad(f, pred), atol=3e-2)


@requires_gurobi
@requires_clarabel
class TestCaVEJax:
    """CaVE cosine-distance gradient cross-checked against torch (projection detached)."""

    def _setup(self):
        from pyepo.model.grb.shortestpath import shortestPathModel

        model = shortestPathModel(grid=(2, 3))  # 7 edges
        d = model.num_cost
        rng = np.random.RandomState(0)
        pred = rng.randn(2, d).astype(np.float32)
        tight = rng.randn(2, 3, d).astype(np.float32)
        return model, pred, tight

    def test_grad_matches_torch(self):
        import jax
        import jax.numpy as jnp
        import torch

        from pyepo.func.cave import coneAlignedCosine as TCaVE
        from pyepo.func.jax import coneAlignedCosine as JCaVE

        model, pred, tight = self._setup()
        jcave = JCaVE(model, reduction="mean")
        g_j = np.array(jax.grad(lambda p: jcave(p, jnp.asarray(tight)))(jnp.asarray(pred)))
        tcave = TCaVE(model, processes=1, reduction="mean")
        pt = torch.tensor(pred, requires_grad=True)
        tcave(pt, torch.as_tensor(tight)).backward()
        np.testing.assert_allclose(g_j, pt.grad.numpy(), atol=1e-3)

    def test_hybrid_heuristic_matches_torch(self):
        import jax
        import jax.numpy as jnp
        import torch

        from pyepo.func.cave import coneAlignedCosine as TCaVE
        from pyepo.func.jax import coneAlignedCosine as JCaVE

        model, pred, tight = self._setup()
        # solve_ratio=0 -> always the cheap heuristic branch (deterministic)
        jcave = JCaVE(model, solve_ratio=0.0, reduction="mean")
        g_j = np.array(jax.grad(lambda p: jcave(p, jnp.asarray(tight)))(jnp.asarray(pred)))
        tcave = TCaVE(model, processes=1, solve_ratio=0.0, reduction="mean")
        pt = torch.tensor(pred, requires_grad=True)
        tcave(pt, torch.as_tensor(tight)).backward()
        np.testing.assert_allclose(g_j, pt.grad.numpy(), atol=1e-3)


@requires_gurobi
class TestMultiprocessing:
    """processes > 1 parallelizes the callback-path solves (results identical to single-core)."""

    def _model(self):
        from pyepo.model.grb.shortestpath import shortestPathModel

        return shortestPathModel(grid=GRID)

    def test_builds_worker_pool(self):
        from pyepo.func.jax import SPOPlus

        assert SPOPlus(self._model(), processes=1).pool is None
        spo = SPOPlus(self._model(), processes=2)
        assert spo.pool is not None

    def test_grad_matches_single_core(self):
        import jax
        import jax.numpy as jnp

        import pyepo
        from pyepo.data.dataset import optDataset
        from pyepo.func.jax import SPOPlus

        x, c = pyepo.data.shortestpath.genData(8, NUM_FEAT, GRID, seed=SEED)
        ds = optDataset(self._model(), x, c)
        pred = (np.asarray(ds.costs) * 1.3).astype(np.float32)
        tc, ts, to = (jnp.asarray(np.asarray(a, np.float32)) for a in (ds.costs, ds.sols, ds.objs))

        def grad_with(n_proc):
            spo = SPOPlus(self._model(), processes=n_proc)
            return np.array(jax.grad(lambda p: spo(p, tc, ts, to))(jnp.asarray(pred)))

        np.testing.assert_allclose(grad_with(2), grad_with(1), atol=1e-5)


@requires_mpax
class TestJitJax:
    def test_spoplus_jit_matches_eager(self):
        """jax.jit of the loss gradient (model closed over) equals the eager gradient."""
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import SPOPlus

        model, ds = _sp_mpax_ds(16)
        pred = (np.asarray(ds.costs) * 1.3).astype(np.float32)
        tcj, tsj, toj = (
            jnp.asarray(np.asarray(a, np.float32)) for a in (ds.costs, ds.sols, ds.objs)
        )
        spo = SPOPlus(model, reduction="mean")

        def loss(p):
            return spo(p, tcj, tsj, toj)

        cj = jnp.asarray(pred)
        g_eager = np.array(jax.grad(loss)(cj))
        g_jit = np.array(jax.jit(jax.grad(loss))(cj))
        np.testing.assert_allclose(g_jit, g_eager, atol=1e-4)


# ============================================================
# A1 hybrid infra: registry-driven contract tests + torch parity
# ============================================================


def _jx(*arrays):
    """torch tensors -> jax arrays."""
    import jax.numpy as jnp

    return tuple(jnp.asarray(a.numpy()) for a in arrays)


@requires_gurobi
class TestJaxContract:
    """Forward/backward contract over JAX_LOSS_REGISTRY, on the Gurobi callback path."""

    @pytest.mark.parametrize("name", JAX_SOLUTION_OPS)
    def test_solution_forward_and_backward(self, name, sp_data):
        import jax
        import jax.numpy as jnp

        optmodel, dataset, loader = sp_data
        _kind, build, sig = JAX_LOSS_REGISTRY[name]
        _x, c, w, z = take_batch(loader)
        cp, cj, wj, zj = _jx(c * 1.2, c, w, z)
        op = build(optmodel, dataset, "mean")
        out = call_op(op, sig, cp, cj, wj, zj)
        assert out.shape == cp.shape
        assert np.isfinite(np.array(out)).all()
        g = jax.grad(lambda p: jnp.sum(call_op(op, sig, p, cj, wj, zj)))(cp)
        assert np.isfinite(np.array(g)).all()

    @pytest.mark.parametrize("name", JAX_LOSS_OPS)
    def test_loss_scalar_and_reduction(self, name, sp_data):
        import jax

        optmodel, dataset, loader = sp_data
        _kind, build, sig = JAX_LOSS_REGISTRY[name]
        _x, c, w, z = take_batch(loader)
        cp, cj, wj, zj = _jx(c * 1.2, c, w, z)
        loss = call_op(build(optmodel, dataset, "mean"), sig, cp, cj, wj, zj)
        assert loss.ndim == 0
        g = jax.grad(lambda p: call_op(build(optmodel, dataset, "mean"), sig, p, cj, wj, zj))(cp)
        assert np.isfinite(np.array(g)).all()
        # reduction modes (fresh seed each build -> deterministic across the three)
        none = call_op(build(optmodel, dataset, "none"), sig, cp, cj, wj, zj)
        assert none.shape[0] == cp.shape[0]
        mean = call_op(build(optmodel, dataset, "mean"), sig, cp, cj, wj, zj)
        total = call_op(build(optmodel, dataset, "sum"), sig, cp, cj, wj, zj)
        np.testing.assert_allclose(float(mean), float(np.array(none).mean()), atol=1e-5)
        np.testing.assert_allclose(float(total), float(np.array(none).sum()), atol=1e-5)


@requires_gurobi
class TestTorchJaxParity:
    """Secondary regression net: jax.grad == torch grad for deterministic losses."""

    @pytest.mark.parametrize("name", DETERMINISTIC_PARITY)
    def test_grad_matches_torch(self, name, sp_data):
        import jax
        import jax.numpy as jnp
        import torch

        optmodel, dataset, loader = sp_data
        _x, c, w, z = take_batch(loader)
        cp_np = (c * 1.2).numpy().astype(np.float32)
        _kind, jbuild, sig = JAX_LOSS_REGISTRY[name]
        _tk, tbuild, _ts = LOSS_REGISTRY[name]
        # torch grad on the same Gurobi model
        top = tbuild(optmodel, dataset, "mean")
        cpt = torch.tensor(cp_np, requires_grad=True)
        out_t = call_op(top, sig, cpt, c, w, z)
        (out_t if out_t.dim() == 0 else out_t.sum()).backward()
        g_t = cpt.grad.numpy()
        # jax grad (Gurobi via callback)
        jop = jbuild(optmodel, dataset, "mean")
        cj, cc, ww, zz = _jx(torch.as_tensor(cp_np), c, w, z)
        g_j = np.array(jax.grad(lambda p: jnp.sum(call_op(jop, sig, p, cc, ww, zz)))(cj))
        np.testing.assert_allclose(g_j, g_t, atol=1e-3)


# ============================================================
# Partial prediction: short predicted cost, full-dimension solution
# ============================================================


def _partial_model():
    """DSL model with 3 predicted-cost vars and 2 fixed-cost vars (num_cost < num_vars)."""
    from pyepo import EPO, dsl

    items = dsl.Variable(3, vtype=EPO.BINARY)
    extra = dsl.Variable(2, vtype=EPO.BINARY)
    cost = dsl.Parameter(3)
    dfix = np.array([1.0, 2.0])
    prob = dsl.Problem(
        dsl.Maximize(cost @ items + dfix @ extra), [items.sum() + extra.sum() <= 3]
    )
    return prob.compile(backend="gurobi")


def _partial_data(n=4):
    from pyepo.data.dataset import optDataset

    model = _partial_model()
    rng = np.random.RandomState(0)
    c = (rng.rand(n, model.num_cost) + 0.5).astype(np.float32)
    x = rng.rand(n, NUM_FEAT).astype(np.float32)
    return model, optDataset(model, x, c), c


@requires_gurobi
class TestPartialPredictionJax:
    """`_full_cost` lift parity for partial prediction (num_cost < num_vars).

    Every other test uses full prediction, where the lift is a no-op, so a
    missing lift in surrogate/perturbed/blackbox/regularized was invisible: the
    short predicted cost would mismatch the full-dimension solution. The lifted
    gradient must still map back to the short cost.
    """

    PARITY = ["SPOPlus", "PG", "DBB", "NID", "RFWO", "RFY"]
    SMOKE = ["DPO", "DPOMul", "IMLE", "AIMLE", "PFY", "PFYMul"]

    @pytest.mark.parametrize("name", PARITY)
    def test_deterministic_grad_matches_torch(self, name):
        import jax
        import jax.numpy as jnp
        import torch

        model, ds, c = _partial_data()
        cn = (c * 1.2).astype(np.float32)
        ct, wt, zt = (np.asarray(a, np.float32) for a in (ds.costs, ds.sols, ds.objs))
        _k, jbuild, sig = JAX_LOSS_REGISTRY[name]
        _tk, tbuild, _ts = LOSS_REGISTRY[name]
        # torch reference (lifts internally via _fullCost)
        top = tbuild(model, ds, "mean")
        cpt = torch.tensor(cn, requires_grad=True)
        out_t = call_op(top, sig, cpt, torch.as_tensor(ct), torch.as_tensor(wt), torch.as_tensor(zt))
        (out_t if out_t.dim() == 0 else out_t.sum()).backward()
        g_t = cpt.grad.numpy()
        # jax grad on the short predicted cost
        jop = jbuild(model, ds, "mean")
        cj, cc, ww, zz = (jnp.asarray(a) for a in (cn, ct, wt, zt))
        g_j = np.array(jax.grad(lambda p: jnp.sum(call_op(jop, sig, p, cc, ww, zz)))(cj))
        assert g_j.shape == cn.shape
        np.testing.assert_allclose(g_j, g_t, atol=1e-3)

    @pytest.mark.parametrize("name", SMOKE)
    def test_perturbed_lifts_and_grad_is_finite(self, name):
        # independent per-framework RNG -> no torch parity; gate is the absence
        # of a shape crash plus a finite gradient on the short predicted cost
        import jax
        import jax.numpy as jnp

        model, ds, c = _partial_data()
        cn = (c * 1.2).astype(np.float32)
        args = (jnp.asarray(np.asarray(a, np.float32)) for a in (ds.costs, ds.sols, ds.objs))
        cc, ww, zz = args
        _k, jbuild, sig = JAX_LOSS_REGISTRY[name]
        jop = jbuild(model, ds, "mean")
        g = np.array(jax.grad(lambda p: jnp.sum(call_op(jop, sig, p, cc, ww, zz)))(jnp.asarray(cn)))
        assert g.shape == cn.shape
        assert np.isfinite(g).all()
