#!/usr/bin/env python
"""Tests for the JAX training frontend."""

import numpy as np
import pytest

from .conftest import (
    GRID,
    JAX_LOSS_REGISTRY,
    LOSS_REGISTRY,
    NUM_FEAT,
    PARTIAL_PREDICTION_PARITY_OPS,
    PARTIAL_PREDICTION_SMOKE_OPS,
    call_op,
    finite_diff_grad,
    requires_clarabel,
    requires_gurobi,
    requires_jax,
    requires_mpax,
    sp_jax_pred,
)

SEED = 42


def _sp_mpax(n):
    """mpax shortest-path model + (n, vars) float32 costs."""
    import pyepo
    from pyepo.model.mpax.shortestpath import shortestPathModel

    _x, c = pyepo.data.shortestpath.genData(n, NUM_FEAT, GRID, seed=SEED)
    return shortestPathModel(grid=GRID), np.asarray(c, np.float32)


# ============================================================
# jax: helpers, infrastructure, validation
# ============================================================


@requires_jax
class TestSolveCacheHelpers:
    """Solution-pool caching helpers (pure jnp, no solver)."""

    @pytest.mark.parametrize(
        "pool, update, expected_rows",
        [
            ([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]], 3),
            ([[1.0, 0.0]], [[1.0 + 1e-6, 1e-6]], 1),
        ],
    )
    def test_update_pool_dedups_and_appends(self, pool, update, expected_rows):
        import jax.numpy as jnp

        from pyepo.func.jax.utils import _update_solution_pool

        out = _update_solution_pool(jnp.array(update), jnp.array(pool))
        assert int(out.shape[0]) == expected_rows

    @pytest.mark.parametrize(
        "sense, expected_sol, expected_obj",
        [
            pytest.param("min", [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [1.0, 1.0]),
            pytest.param("max", [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], [3.0, 3.0]),
        ],
    )
    def test_cache_in_pass_picks_best_obj(self, sense, expected_sol, expected_obj):
        from unittest.mock import MagicMock

        import jax.numpy as jnp

        from pyepo import EPO
        from pyepo.func.jax.utils import _cache_in_pass

        m = MagicMock()
        m.modelSense = EPO.MINIMIZE if sense == "min" else EPO.MAXIMIZE
        cost = jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        pool = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        sol, obj = _cache_in_pass(cost, m, pool)
        np.testing.assert_allclose(np.array(sol), expected_sol)
        np.testing.assert_allclose(np.array(obj), expected_obj)

    def test_cache_in_pass_rejects_invalid_sense(self):
        from unittest.mock import MagicMock

        import jax.numpy as jnp

        from pyepo.func.jax.utils import _cache_in_pass

        model = MagicMock()
        model.modelSense = "invalid"
        with pytest.raises(ValueError, match="Invalid modelSense"):
            _cache_in_pass(jnp.ones((1, 2)), model, jnp.eye(2))

    @requires_mpax  # the eager caching pass solves through the native MPAX path
    def test_spoplus_caching_runs_eager(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import SPOPlus

        model, ds, _pred, tc, ts, to = sp_jax_pred("mpax", 16)
        pred = (np.random.RandomState(0).rand(*tc.shape) + 0.1).astype(np.float32)
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


@requires_jax
class TestFrankWolfeFreeSlot:
    """Active-set slot selection."""

    def test_matches_expected_slots(self):
        import jax.numpy as jnp

        from pyepo.func.jax.regularized import _fw_free_slot

        w = jnp.asarray([[0.5, 0.0, 0.5, 0.0], [0.4, 0.3, 0.1, 0.2]])
        np.testing.assert_array_equal(np.array(_fw_free_slot(w)), [1, 2])


@requires_jax
class TestMaskPred:
    """Partial-prediction perturbation mask."""

    @pytest.mark.parametrize(
        "c_pred_index, expected",
        [
            (None, np.ones((1, 2, 4))),
            (np.array([0, 2]), np.array([[[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]])),
        ],
    )
    def test_masks_fixed_positions(self, c_pred_index, expected):
        from unittest.mock import MagicMock

        import jax.numpy as jnp

        from pyepo.func.jax.utils import _mask_pred

        m = MagicMock()
        m.c_pred_index = c_pred_index
        noises = jnp.ones((1, 2, 4))
        np.testing.assert_array_equal(np.array(_mask_pred(noises, m)), expected)


# optModule init validation is shared with the torch frontend in
# test_50_func.py (the frontend-parametrized TestOptModuleInit /
# TestConstructorGuards).


@requires_mpax
class TestBatchSolve:
    def test_callback_matches_native(self):
        import jax.numpy as jnp

        from pyepo.func.jax.utils import _solve_batch_callback, _solve_batch_mpax

        model, c = _sp_mpax(8)
        c_jax = jnp.asarray(c)
        sol_n, obj_n = _solve_batch_mpax(c_jax, model)
        sol_c, obj_c = _solve_batch_callback(c_jax, model)
        # 1e-2: two independent first-order PDHG solves of the same costs
        np.testing.assert_allclose(np.array(sol_c), np.array(sol_n), atol=1e-2)
        np.testing.assert_allclose(np.array(obj_c), np.array(obj_n), atol=1e-2)


@requires_jax
@requires_gurobi
class TestMultiprocessing:
    """Callback-path multiprocessing."""

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
class TestJit:
    def test_spoplus_jit_matches_eager(self):
        """jit gradient matches eager."""
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import SPOPlus

        model, _ds, pred, tc, ts, to = sp_jax_pred("mpax", 16)
        tcj, tsj, toj = (jnp.asarray(a) for a in (tc, ts, to))
        spo = SPOPlus(model, reduction="mean")

        def loss(p):
            return spo(p, tcj, tsj, toj)

        cj = jnp.asarray(pred)
        g_eager = np.array(jax.grad(loss)(cj))
        g_jit = np.array(jax.jit(jax.grad(loss))(cj))
        np.testing.assert_allclose(g_jit, g_eager, atol=1e-4)

    def test_jit_caching_raises_clear_error(self):
        """Caching loss rejects jit."""
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import noiseContrastiveEstimation

        model, ds, _pred, tc, ts_np, _to = sp_jax_pred("mpax", 12)
        ts = jnp.asarray(ts_np)
        pred = jnp.asarray((tc * 1.2).astype(np.float32))
        nce = noiseContrastiveEstimation(model, dataset=ds, solve_ratio=0.0)
        # eager is fine
        assert np.isfinite(np.array(jax.grad(lambda p: nce(p, ts))(pred))).all()
        # jit raises a clear, actionable error
        with pytest.raises(RuntimeError, match=r"jax\.jit"):
            jax.jit(jax.grad(lambda p: nce(p, ts)))(pred)


@requires_jax
@requires_gurobi
class TestJitKeyGuard:
    """JIT requires explicit RNG keys."""

    def test_perturbed_jit_requires_explicit_key(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import perturbedOpt

        model, c = _perturbed_setup("max")
        dpo = perturbedOpt(model, n_samples=3, sigma=1.0, seed=0)
        pred = jnp.asarray((c * 1.2).astype(np.float32))
        with pytest.raises(RuntimeError, match="key"):
            jax.jit(lambda p: dpo(p).sum())(pred)
        # an explicit key is a traced argument: jit matches eager
        key = jax.random.PRNGKey(7)
        out_jit = jax.jit(lambda p, k: dpo(p, key=k))(pred, key)
        np.testing.assert_allclose(np.array(out_jit), np.array(dpo(pred, key=key)), atol=1e-6)


# ============================================================
# jax: independent correctness gates
# ============================================================


@requires_jax
@requires_gurobi
class TestPGLabelGradient:
    """PG: the true-cost label carries no gradient."""

    def test_true_cost_grad_is_zero(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import PG

        model, c = _perturbed_setup("max")
        pg = PG(model, sigma=0.1)
        pred = jnp.asarray((c * 1.2).astype(np.float32))
        g = jax.grad(lambda p, t: pg(p, t), argnums=1)(pred, jnp.asarray(c))
        np.testing.assert_allclose(np.array(g), 0.0, atol=1e-7)


def _spo_closed_form(model, pred, true_cost, true_sol):
    """Independent ground truth: 2*(w_true - w_spo) for MINIMIZE."""
    import jax.numpy as jnp

    from pyepo.func.jax.utils import _solve_batch

    w_spo, _ = _solve_batch(jnp.asarray(2.0 * pred - true_cost), model)
    return 2.0 * (true_sol - np.array(w_spo))


# SPO+ closed form on native MPAX and Gurobi callback.
SPO_CLOSED_FORM = [
    pytest.param("mpax", 16, 1e-2, marks=requires_mpax),
    pytest.param("grb", 8, 1e-4, marks=requires_gurobi),
]


@requires_jax
class TestSPOPlusClosedForm:
    """SPO+ closed-form subgradient."""

    @pytest.mark.parametrize("backend,n,atol", SPO_CLOSED_FORM)
    def test_grad_matches_closed_form(self, backend, n, atol):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import SPOPlus

        model, _ds, pred, tc, ts, to = sp_jax_pred(backend, n)
        B = pred.shape[0]
        spo = SPOPlus(model, reduction="mean")
        grad = np.array(
            jax.grad(lambda p: spo(p, jnp.asarray(tc), jnp.asarray(ts), jnp.asarray(to)))(
                jnp.asarray(pred)
            )
        )
        expected = _spo_closed_form(model, pred, tc, ts) / B
        np.testing.assert_allclose(grad, expected, atol=atol)


@requires_mpax
class TestBlackbox:
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
        from pyepo.func.jax.utils import _solve_batch

        model, pred, target = self._setup()
        lambd = 10.0
        dbb = blackboxOpt(model, lambd=lambd)
        g = np.array(jax.grad(lambda p: jnp.sum(jnp.asarray(target) * dbb(p)))(jnp.asarray(pred)))
        # closed form: (w*(pred + lambd*d) - w*(pred)) / lambd with d = target
        wp, _ = _solve_batch(jnp.asarray(pred), model)
        wq, _ = _solve_batch(jnp.asarray(pred + lambd * target), model)
        expected = (np.array(wq) - np.array(wp)) / lambd
        np.testing.assert_allclose(g, expected, atol=1e-2)  # mpax PDHG re-solve


def _perturbed_setup(sense):
    """Model and base costs for MINIMIZE or MAXIMIZE tests."""
    import pyepo

    if sense == "min":
        from pyepo.model.mpax.shortestpath import shortestPathModel

        _x, c = pyepo.data.shortestpath.genData(8, NUM_FEAT, GRID, seed=SEED)
        return shortestPathModel(grid=GRID), np.asarray(c, np.float32)
    from pyepo.model.grb.knapsack import knapsackModel

    model = knapsackModel(weights=[[3.0, 4.0, 2.0, 5.0, 3.0]], capacity=[10.0])
    c = (np.random.RandomState(0).rand(8, model.num_cost) + 0.5).astype(np.float32)
    return model, c


# MINIMIZE on MPAX, MAXIMIZE on Gurobi.
PERTURBED_SENSE = [
    pytest.param("min", marks=requires_mpax),
    pytest.param("max", marks=requires_gurobi),
]


@requires_jax
class TestPerturbed:
    """DPO/PFY gradients from seed-derived noise."""

    @staticmethod
    def _gauss(shape):
        import jax
        import jax.numpy as jnp

        _key, sub = jax.random.split(jax.random.PRNGKey(0))
        return jax.random.normal(sub, shape, dtype=jnp.float32)

    @staticmethod
    def _perturb(pred, noises, sigma, mul):
        # Match the implementation path exactly.
        import jax.numpy as jnp

        p = jnp.asarray(pred)[:, None, :]
        if mul:
            return p * jnp.exp(sigma * noises - 0.5 * sigma**2)
        return p + sigma * noises

    @pytest.mark.parametrize("sense", PERTURBED_SENSE)
    @pytest.mark.parametrize("mul", [False, True])
    def test_perturbed_opt_grad_matches_reference(self, sense, mul):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import perturbedOpt, perturbedOptMul
        from pyepo.func.jax.utils import _solve_batch

        model, c = _perturbed_setup(sense)
        sigma = 0.5 if mul else 1.0
        pred = (c * 1.3).astype(np.float32)
        B, d = pred.shape
        target = np.random.RandomState(3).randn(B, d).astype(np.float32)
        noises = self._gauss((B, 5, d))
        ptb_c = self._perturb(pred, noises, sigma, mul)
        sols, _ = _solve_batch(jnp.asarray(ptb_c.reshape(-1, d)), model)
        ptb_sols = np.array(sols).reshape(B, 5, d)
        reward = np.einsum("bnd,bd->bn", ptb_sols, target)
        reward = (reward - reward.mean(1, keepdims=True)) * (5 / 4)
        denom = 5 * sigma * pred if mul else 5 * sigma
        expected = np.einsum("bnd,bn->bd", np.array(noises), reward) / denom
        cls = perturbedOptMul if mul else perturbedOpt
        po = cls(model, n_samples=5, sigma=sigma, seed=0)
        g = np.array(jax.grad(lambda p: jnp.sum(jnp.asarray(target) * po(p)))(jnp.asarray(pred)))
        atol = 5e-2 if sense == "min" else 1e-3  # min runs on mpax (PDHG re-solve)
        np.testing.assert_allclose(g, expected, atol=atol)

    @pytest.mark.parametrize("sense", PERTURBED_SENSE)
    @pytest.mark.parametrize("mul", [False, True])
    def test_perturbed_fenchel_young_grad_matches_reference(self, sense, mul):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import perturbedFenchelYoung, perturbedFenchelYoungMul
        from pyepo.func.jax.utils import _solve_batch

        model, c = _perturbed_setup(sense)
        sigma = 0.5 if mul else 1.0
        w_true = np.array(_solve_batch(jnp.asarray(c), model)[0])
        pred = (c * 1.3).astype(np.float32)
        B, d = pred.shape
        noises = self._gauss((B, 5, d))
        ptb_c = self._perturb(pred, noises, sigma, mul)
        sols, _ = _solve_batch(jnp.asarray(ptb_c.reshape(-1, d)), model)
        ptb_sols = np.array(sols).reshape(B, 5, d)
        if mul:
            factor = np.array(jnp.exp(sigma * noises - 0.5 * sigma**2))
            e_sol = (ptb_sols * factor).mean(1)
        else:
            e_sol = ptb_sols.mean(1)
        # Fenchel-Young residual: w - e_sol (MIN), e_sol - w (MAX)
        diff = (e_sol - w_true) if sense == "max" else (w_true - e_sol)
        expected = diff / B
        cls = perturbedFenchelYoungMul if mul else perturbedFenchelYoung
        pfy = cls(model, n_samples=5, sigma=sigma, seed=0)
        g = np.array(jax.grad(lambda p: pfy(p, jnp.asarray(w_true)))(jnp.asarray(pred)))
        atol = 5e-2 if sense == "min" else 1e-3  # min runs on mpax (PDHG re-solve)
        np.testing.assert_allclose(g, expected, atol=atol)


@requires_mpax
class TestImplicitMLE:
    """I-MLE / AI-MLE gradients."""

    def _setup(self):
        """Model, costs, target, and perturbed costs."""
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax.utils import _sum_gamma_sample

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

        from pyepo.func.jax.perturbed import _solve_or_cache_3d

        ptb_sols = np.array(_solve_or_cache_3d(ptb_c, module))
        sols_pos = np.array(
            _solve_or_cache_3d(ptb_c + lambd * jnp.asarray(target)[:, None, :], module)
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
        np.testing.assert_allclose(g, expected, atol=5e-2)  # mpax PDHG re-solve

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
        np.testing.assert_allclose(g, expected, atol=5e-2)  # mpax PDHG re-solve
        assert aimle.alpha != a0

    @staticmethod
    def _resolve_grad_two_sides(module, target, ptb_c, lambd):
        import jax.numpy as jnp

        from pyepo.func.jax.perturbed import _solve_or_cache_3d

        delta = lambd * jnp.asarray(target)[:, None, :]
        pos = np.array(_solve_or_cache_3d(ptb_c + delta, module))
        neg = np.array(_solve_or_cache_3d(ptb_c - delta, module))
        return (pos - neg).mean(axis=1) / (2 * lambd)

    def test_implicit_mle_two_sides_grad_matches_reference(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import implicitMLE

        model, pred, target, ptb_c = self._setup()
        lambd = 10.0
        imle = implicitMLE(
            model, n_samples=5, sigma=1.0, lambd=lambd, kappa=5.0, two_sides=True, seed=0
        )
        expected = self._resolve_grad_two_sides(imle, target, ptb_c, lambd)
        g = np.array(jax.grad(lambda p: jnp.sum(jnp.asarray(target) * imle(p)))(jnp.asarray(pred)))
        np.testing.assert_allclose(g, expected, atol=5e-2)  # mpax PDHG re-solve

    def test_adaptive_two_sides_grad_matches_reference(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import adaptiveImplicitMLE

        model, pred, target, ptb_c = self._setup()
        aimle = adaptiveImplicitMLE(
            model, n_samples=5, sigma=1.0, kappa=5.0, two_sides=True, seed=0
        )
        lambd = aimle.alpha * float(np.linalg.norm(pred)) / float(np.linalg.norm(target))
        expected = self._resolve_grad_two_sides(aimle, target, ptb_c, lambd)
        g = np.array(jax.grad(lambda p: jnp.sum(jnp.asarray(target) * aimle(p)))(jnp.asarray(pred)))
        np.testing.assert_allclose(g, expected, atol=5e-2)  # mpax PDHG re-solve


@requires_jax
@requires_gurobi
class TestMaximizeSenseEstimators:
    """MAXIMIZE estimator gradients."""

    def _setup(self):
        model, c = _perturbed_setup("max")
        pred = (c * 1.2).astype(np.float32)
        target = np.random.RandomState(3).randn(*pred.shape).astype(np.float32)
        return model, pred, target

    def test_blackbox_opt_grad_matches_closed_form(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import blackboxOpt
        from pyepo.func.jax.utils import _solve_batch

        model, pred, target = self._setup()
        lambd = 10.0
        dbb = blackboxOpt(model, lambd=lambd)
        g = np.array(jax.grad(lambda p: jnp.sum(jnp.asarray(target) * dbb(p)))(jnp.asarray(pred)))
        # MAXIMIZE: perturb against the upstream gradient and flip the sign
        wp, _ = _solve_batch(jnp.asarray(pred), model)
        wq, _ = _solve_batch(jnp.asarray(pred - lambd * target), model)
        expected = -(np.array(wq) - np.array(wp)) / lambd
        np.testing.assert_allclose(g, expected, atol=1e-4)

    def test_implicit_mle_grad_matches_reference(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import implicitMLE
        from pyepo.func.jax.perturbed import _solve_or_cache_3d
        from pyepo.func.jax.utils import _sum_gamma_sample

        model, pred, target = self._setup()
        lambd = 10.0
        imle = implicitMLE(model, n_samples=5, sigma=1.0, lambd=lambd, kappa=5.0, seed=0)
        # seed-derived noise reproduces the module's perturbed costs
        b, d = pred.shape
        _key, sub = jax.random.split(jax.random.PRNGKey(0))
        noises = _sum_gamma_sample(sub, 5.0, 10, (b, 5, d))
        ptb_c = jnp.asarray(pred)[:, None, :] + 1.0 * noises
        g = np.array(jax.grad(lambda p: jnp.sum(jnp.asarray(target) * imle(p)))(jnp.asarray(pred)))
        # MAXIMIZE: perturb against the upstream gradient and flip the sign
        ptb_sols = np.array(_solve_or_cache_3d(ptb_c, imle))
        delta = lambd * jnp.asarray(target)[:, None, :]
        sols_neg = np.array(_solve_or_cache_3d(ptb_c - delta, imle))
        expected = -(sols_neg - ptb_sols).mean(axis=1) / lambd
        np.testing.assert_allclose(g, expected, atol=1e-4)


@requires_mpax
class TestPG:
    """PG backward-difference gradient."""

    def test_grad_matches_backward_difference(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import PG
        from pyepo.func.jax.utils import _solve_batch
        from pyepo.utils import _EPS

        model, _ds, pred, tc, _ts, _to = sp_jax_pred("mpax", 8)
        sigma = 1.0
        B = pred.shape[0]
        pg = PG(model, sigma=sigma, reduction="mean")
        g = np.array(jax.grad(lambda p: pg(p, jnp.asarray(tc)))(jnp.asarray(pred)))
        w_sol, _ = _solve_batch(jnp.asarray(pred), model)
        wm_sol, _ = _solve_batch(jnp.asarray(pred - sigma * tc), model)
        expected = (np.array(w_sol) - np.array(wm_sol)) / (sigma + _EPS) / B  # sign +1 (MINIMIZE)
        np.testing.assert_allclose(g, expected, atol=1e-2)  # mpax PDHG re-solve


@requires_jax
@requires_gurobi
class TestRegularized:
    """Regularized FW gradients."""

    def _knapsack(self):
        from pyepo.model.grb.knapsack import knapsackModel

        return knapsackModel(weights=[[3.0, 4.0, 2.0, 5.0]], capacity=[7.0])

    def test_opt_grad_matches_finite_difference(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import regularizedFrankWolfeOpt as JOpt

        lambd = 1.0
        rng = np.random.RandomState(0)
        cp = (rng.rand(1, 4) * 2 + 1.0).astype(np.float32)
        target = rng.randn(1, 4).astype(np.float32)
        opt = JOpt(self._knapsack(), lambd=lambd, max_iter=30, tol=1e-8)
        g = np.array(jax.grad(lambda p: jnp.sum(jnp.asarray(target) * opt(p)))(jnp.asarray(cp)))

        def value(c):
            return float(jnp.sum(jnp.asarray(target) * opt(jnp.asarray(c))))

        fd = finite_diff_grad(value, cp, eps=5e-3)
        np.testing.assert_allclose(g, fd, atol=5e-2)

    def test_fy_grad_matches_danskin_residual(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import regularizedFrankWolfeFenchelYoung
        from pyepo.func.jax.regularized import _away_step_frank_wolfe

        model = self._knapsack()
        lambd = 1.0
        cp = np.array([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]], np.float32)
        w = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]], np.float32)
        B = cp.shape[0]
        fy = regularizedFrankWolfeFenchelYoung(model, lambd=lambd, max_iter=100, tol=1e-8)
        g = np.array(jax.grad(lambda p: fy(p, jnp.asarray(w)))(jnp.asarray(cp)))
        # MAXIMIZE: theta = pred/lambd, Danskin residual diff = r_sol - w
        r_sol = np.array(_away_step_frank_wolfe(jnp.asarray(cp) / lambd, fy)[0])
        expected = (r_sol - w) / B
        np.testing.assert_allclose(g, expected, atol=1e-3)

    def test_caching_reads_pool_and_grows(self):
        # Cached passes read the pool; exact passes may grow it.
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


@requires_mpax
class TestRankContrastive:
    """Pool and contrastive loss gradients."""

    def _data(self):
        model, ds, pred, tc, ts, _to = sp_jax_pred("mpax", 12)
        return model, ds, pred, tc, ts

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
        # Freeze the pool for deterministic finite difference.
        loss = getattr(J, name)(model, dataset=ds, solve_ratio=0)
        second = jnp.asarray(true_cost if arg == "cost" else true_sol)

        def f(p):
            return float(loss(jnp.asarray(p), second))

        g_auto = np.array(jax.grad(lambda p: loss(p, second))(jnp.asarray(pred)))
        np.testing.assert_allclose(g_auto, finite_diff_grad(f, pred), atol=3e-2)


# ============================================================
# both: Torch-vs-JAX parity
# ============================================================


@requires_jax
class TestListwiseLTRParity:
    """Listwise LTR Torch parity."""

    @pytest.mark.parametrize("sense", PERTURBED_SENSE)
    def test_loss_matches_torch(self, sense):
        import jax.numpy as jnp
        import torch

        from pyepo.data.dataset import optDataset
        from pyepo.func.jax import listwiseLearningToRank as JLTR
        from pyepo.func.rank import listwiseLearningToRank as TLTR

        model, c = _perturbed_setup(sense)
        feats = np.random.RandomState(1).rand(len(c), NUM_FEAT).astype(np.float32)
        ds = optDataset(model, feats, c)
        pred = (c * 1.2).astype(np.float32)
        # solve_ratio=0 freezes both pools at the identical dataset seed
        t_loss = TLTR(model, solve_ratio=0, dataset=ds)(
            torch.as_tensor(pred), torch.as_tensor(c, dtype=torch.float32)
        )
        j_loss = JLTR(model, solve_ratio=0, dataset=ds)(jnp.asarray(pred), jnp.asarray(c))
        np.testing.assert_allclose(float(j_loss), t_loss.item(), atol=1e-5)


@requires_jax
@requires_gurobi
class TestPGTwoSidesParity:
    """PG central-difference Torch parity."""

    def _model(self, sense):
        if sense == "min":
            from pyepo.model.grb.shortestpath import shortestPathModel

            return shortestPathModel(grid=GRID)
        from pyepo.model.grb.knapsack import knapsackModel

        return knapsackModel(weights=[[3.0, 4.0, 2.0, 5.0, 3.0]], capacity=[10.0])

    @pytest.mark.parametrize("sense", ["min", "max"])
    def test_two_sides_grad_matches_torch(self, sense):
        import jax
        import jax.numpy as jnp
        import torch

        from pyepo.func import PG as TPG
        from pyepo.func.jax import PG as JPG

        model = self._model(sense)
        d = model.num_cost
        rng = np.random.RandomState(0)
        c = (rng.rand(4, d) + 0.5).astype(np.float32)
        cp = (c * 1.2).astype(np.float32)
        # torch reference (deterministic: no internal noise)
        tpg = TPG(model, sigma=0.5, two_sides=True, processes=1, reduction="mean")
        cpt = torch.tensor(cp, requires_grad=True)
        tpg(cpt, torch.as_tensor(c)).backward()
        g_t = cpt.grad.numpy()
        # jax
        jpg = JPG(model, sigma=0.5, two_sides=True, reduction="mean")
        g_j = np.array(jax.grad(lambda p: jpg(p, jnp.asarray(c)))(jnp.asarray(cp)))
        np.testing.assert_allclose(g_j, g_t, atol=1e-3)


@requires_jax
@requires_gurobi
@requires_clarabel
class TestCaVEParity:
    """CaVE Torch parity."""

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


@requires_jax
@requires_gurobi
@requires_clarabel
class TestCaVEGuards:
    """CaVE: detached labels and the jit guard on the hybrid coin."""

    def _setup(self):
        from pyepo.model.grb.shortestpath import shortestPathModel

        model = shortestPathModel(grid=(2, 3))  # 7 edges
        rng = np.random.RandomState(0)
        pred = rng.randn(2, model.num_cost).astype(np.float32)
        tight = rng.randn(2, 3, model.num_cost).astype(np.float32)
        return model, pred, tight

    def test_tight_ctrs_grad_is_zero(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import coneAlignedCosine as JCaVE

        model, pred, tight = self._setup()
        jcave = JCaVE(model, reduction="mean")
        # labels carry no gradient
        g = jax.grad(lambda t: jcave(jnp.asarray(pred), t), argnums=0)(jnp.asarray(tight))
        np.testing.assert_allclose(np.array(g), 0.0, atol=1e-7)

    def test_hybrid_coin_rejected_under_jit(self):
        import jax
        import jax.numpy as jnp

        from pyepo.func.jax import coneAlignedCosine as JCaVE

        model, pred, tight = self._setup()
        jcave = JCaVE(model, solve_ratio=0.5, reduction="mean")
        with pytest.raises(RuntimeError, match="jit"):
            jax.jit(lambda p: jcave(p, jnp.asarray(tight)))(jnp.asarray(pred))


def _partial_model():
    """Partial-prediction DSL model."""
    from pyepo import EPO, dsl

    items = dsl.Variable(3, vtype=EPO.BINARY)
    extra = dsl.Variable(2, vtype=EPO.BINARY)
    cost = dsl.Parameter(3)
    dfix = np.array([1.0, 2.0])
    prob = dsl.Problem(dsl.Maximize(cost @ items + dfix @ extra), [items.sum() + extra.sum() <= 3])
    return prob.compile(backend="gurobi")


def _partial_data(n=4):
    from pyepo.data.dataset import optDataset

    model = _partial_model()
    rng = np.random.RandomState(0)
    c = (rng.rand(n, model.num_cost) + 0.5).astype(np.float32)
    x = rng.rand(n, NUM_FEAT).astype(np.float32)
    return model, optDataset(model, x, c), c


@requires_jax
@requires_gurobi
class TestPartialPredictionParity:
    """Partial-prediction lift parity."""

    def test_multiplicative_perturb_keeps_fixed_costs(self):
        import jax.numpy as jnp

        from pyepo.func.jax.perturbed import _perturb
        from pyepo.func.jax.utils import _full_cost, _mask_pred

        model = _partial_model()
        full = _full_cost(jnp.ones((2, model.num_cost)), model)
        raw = np.random.RandomState(0).randn(2, 3, full.shape[-1]).astype(np.float32)
        noises = _mask_pred(jnp.asarray(raw), model)
        ptb_c = np.array(_perturb(full, noises, 1.0, True, model))
        # non-predicted positions keep the known fixed cost exactly
        fixed = np.setdiff1d(np.arange(full.shape[-1]), np.asarray(model.c_pred_index))
        expected = np.broadcast_to(np.array(full)[:, None, fixed], (2, 3, fixed.size))
        np.testing.assert_allclose(ptb_c[:, :, fixed], expected, atol=1e-7)

    @pytest.mark.parametrize("name", PARTIAL_PREDICTION_PARITY_OPS)
    def test_deterministic_grad_matches_torch(self, name):
        import jax
        import jax.numpy as jnp
        import torch

        model, ds, c = _partial_data()
        cn = (c * 1.2).astype(np.float32)
        ct, wt, zt = (np.asarray(a, np.float32) for a in (ds.costs, ds.sols, ds.objs))
        jentry = JAX_LOSS_REGISTRY[name]
        tentry = LOSS_REGISTRY[name]
        # Torch reference for lifted full costs
        top = tentry.build(model, ds, "mean")
        cpt = torch.tensor(cn, requires_grad=True)
        out_t = call_op(
            top,
            jentry.sig,
            cpt,
            torch.as_tensor(ct),
            torch.as_tensor(wt),
            torch.as_tensor(zt),
        )
        (out_t if out_t.dim() == 0 else out_t.sum()).backward()
        g_t = cpt.grad.numpy()
        # JAX gradient on the short cost
        jop = jentry.build(model, ds, "mean")
        cj, cc, ww, zz = (jnp.asarray(a) for a in (cn, ct, wt, zt))
        g_j = np.array(jax.grad(lambda p: jnp.sum(call_op(jop, jentry.sig, p, cc, ww, zz)))(cj))
        assert g_j.shape == cn.shape
        np.testing.assert_allclose(g_j, g_t, atol=1e-3)

    @pytest.mark.parametrize("name", PARTIAL_PREDICTION_SMOKE_OPS)
    def test_perturbed_lifts_and_grad_is_finite(self, name):
        # Smoke-test perturbed partial prediction
        import jax
        import jax.numpy as jnp

        model, ds, c = _partial_data()
        cn = (c * 1.2).astype(np.float32)
        args = (jnp.asarray(np.asarray(a, np.float32)) for a in (ds.costs, ds.sols, ds.objs))
        cc, ww, zz = args
        entry = JAX_LOSS_REGISTRY[name]
        jop = entry.build(model, ds, "mean")
        g = np.array(
            jax.grad(lambda p: jnp.sum(call_op(jop, entry.sig, p, cc, ww, zz)))(jnp.asarray(cn))
        )
        assert g.shape == cn.shape
        assert np.isfinite(g).all()


@requires_jax
@requires_gurobi
def test_full_prediction_lift_includes_offset():
    """Full-prediction lift includes fixed offsets."""
    import jax.numpy as jnp
    import torch

    from pyepo import EPO, dsl
    from pyepo.func.jax.utils import _full_cost

    x = dsl.Variable(4, vtype=EPO.BINARY)
    c = dsl.Parameter(4)
    d = np.array([1.0, 2.0, 3.0, 4.0])
    model = dsl.Problem(dsl.Minimize((c + d) @ x), [x.sum() >= 1]).compile(backend="gurobi")
    pred = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    jax_full = np.asarray(_full_cost(jnp.asarray(pred), model))
    torch_full = model._fullCost(torch.as_tensor(pred)).numpy()
    np.testing.assert_allclose(jax_full, torch_full)
    np.testing.assert_allclose(jax_full, pred + d)
