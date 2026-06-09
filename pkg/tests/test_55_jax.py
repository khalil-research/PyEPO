#!/usr/bin/env python
"""Tests for the JAX training frontend (pyepo.func.jax).

Correctness gate is the closed form (independent re-solve), not torch-vs-jax.
Covers both backends of batch_solve: the universal pure_callback path and the
native MPAX path. A jax-only training step confirms end-to-end gradient flow.
"""

import numpy as np

from .conftest import GRID, NUM_FEAT, requires_gurobi, requires_mpax

SEED = 42


@requires_mpax
class TestBatchSolve:

    def _model_costs(self):
        import jax.numpy as jnp

        import pyepo
        from pyepo.model.mpax.shortestpath import shortestPathModel

        _x, c = pyepo.data.shortestpath.genData(8, NUM_FEAT, GRID, seed=SEED)
        model = shortestPathModel(grid=GRID)
        c32 = np.asarray(c, dtype=np.float32)
        return model, jnp.asarray(c32), c32

    def test_callback_matches_native(self):
        from pyepo.func.jax.solve import _batch_solve_callback, _batch_solve_mpax

        model, c_jax, _ = self._model_costs()
        sol_n, obj_n = _batch_solve_mpax(c_jax, model)
        sol_c, obj_c = _batch_solve_callback(c_jax, model)
        np.testing.assert_allclose(np.array(sol_c), np.array(sol_n), atol=1e-3)
        np.testing.assert_allclose(np.array(obj_c), np.array(obj_n), atol=1e-3)


def _spo_closed_form(model, pred, true_cost, true_sol):
    """Independent ground truth: 2*(w_true - w_spo) for MINIMIZE."""
    import jax.numpy as jnp

    from pyepo.func.jax.solve import batch_solve

    w_spo, _ = batch_solve(jnp.asarray(2.0 * pred - true_cost), model)
    return 2.0 * (true_sol - np.array(w_spo))


@requires_mpax
class TestSPOPlusJaxMpax:

    def _setup(self):
        import pyepo
        from pyepo.data.dataset import optDataset
        from pyepo.model.mpax.shortestpath import shortestPathModel

        x, c = pyepo.data.shortestpath.genData(16, NUM_FEAT, GRID, seed=SEED)
        model = shortestPathModel(grid=GRID)
        ds = optDataset(model, x, c)
        pred = (np.asarray(ds.costs) * 1.3).astype(np.float32)
        return (model,
                pred,
                np.asarray(ds.costs, np.float32),
                np.asarray(ds.sols, np.float32),
                np.asarray(ds.objs, np.float32))

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
        params = {"W": jnp.asarray(0.01 * rng.randn(NUM_FEAT, tc.shape[1]).astype(np.float32)),
                  "b": jnp.zeros((tc.shape[1],), jnp.float32)}
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

        from pyepo.func.jax.solve import _update_solution_pool

        pool = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        # one duplicate row, one new row
        out = _update_solution_pool(jnp.array([[1.0, 0.0], [1.0, 1.0]]), pool)
        assert int(out.shape[0]) == 3

    def test_cache_in_pass_minimize_picks_min_obj(self):
        from unittest.mock import MagicMock

        import jax.numpy as jnp

        from pyepo import EPO
        from pyepo.func.jax.solve import _cache_in_pass

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

        import pyepo
        from pyepo.data.dataset import optDataset
        from pyepo.func.jax import SPOPlus
        from pyepo.model.mpax.shortestpath import shortestPathModel

        x, c = pyepo.data.shortestpath.genData(16, NUM_FEAT, GRID, seed=SEED)
        model = shortestPathModel(grid=GRID)
        ds = optDataset(model, x, c)
        pred = (np.random.RandomState(0).rand(*np.asarray(ds.costs).shape) + 0.1).astype(np.float32)
        tc, ts, to = (np.asarray(a, np.float32) for a in (ds.costs, ds.sols, ds.objs))
        spo = SPOPlus(model, solve_ratio=0.5, dataset=ds)
        # force the solve-and-grow branch deterministically
        spo.solve_ratio = 1.0
        n0 = int(spo.solpool.shape[0])
        grad = np.array(jax.grad(
            lambda p: spo(p, jnp.asarray(tc), jnp.asarray(ts), jnp.asarray(to)))(jnp.asarray(pred)))
        assert np.isfinite(grad).all()
        assert int(spo.solpool.shape[0]) >= n0
