#!/usr/bin/env python
"""
Utility functions for the JAX frontend
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pyepo import EPO
from pyepo.func.utils import _solve_batch
from pyepo.model.mpax import optMpaxModel

try:
    from jax.extend.core import concrete_or_error as _concrete_or_error
except ImportError:  # older jax
    from jax.core import concrete_or_error as _concrete_or_error


def _check_jit_caching(cost, module):
    """Raise a clear error if a pool-caching side effect is traced under jax.jit."""
    if module.solpool is None:
        return
    try:
        _concrete_or_error(None, cost)
    except jax.errors.ConcretizationTypeError:
        raise RuntimeError(
            "JAX pool caching is not supported under jax.jit; run the loss eagerly "
            "(call jax.grad without wrapping it in jax.jit)."
        ) from None


def batch_solve(cost, optmodel, processes=1, pool=None):
    """
    A function to solve a batch of costs for the JAX frontend
    """
    # MPAX solves natively; any other backend via pure_callback
    if isinstance(optmodel, optMpaxModel):
        return _batch_solve_mpax(cost, optmodel)
    return _batch_solve_callback(cost, optmodel, processes, pool)


def _batch_solve_mpax(cost, optmodel):
    """
    A function to solve natively with MPAX
    """
    # change sign for maximization
    cc = -cost if optmodel.modelSense == EPO.MAXIMIZE else cost
    # batch solving
    sol, obj = optmodel.batch_optimize(cc)
    # obj in true sense
    if optmodel.modelSense == EPO.MAXIMIZE:
        obj = -obj
    return sol, obj


def _batch_solve_callback(cost, optmodel, processes=1, pool=None):
    """
    A function to solve via jax.pure_callback over _solve_batch
    """
    b, n = cost.shape

    def _np_solve(c_np):
        # solve on host
        sol, obj = _solve_batch(np.asarray(c_np, dtype=np.float32), optmodel, processes, pool)
        # to numpy
        return (
            np.asarray(sol.detach().cpu(), dtype=np.float32),
            np.asarray(obj.detach().cpu(), dtype=np.float32),
        )

    out = (
        jax.ShapeDtypeStruct((b, n), jnp.float32),
        jax.ShapeDtypeStruct((b,), jnp.float32),
    )
    return jax.pure_callback(_np_solve, out, cost)


def solve_or_cache(cost, module):
    """
    A function to solve a batch or reuse the solution pool
    """
    _check_jit_caching(cost, module)
    if module._branch_rng.uniform() <= module.solve_ratio:
        sol, obj = batch_solve(cost, module.optmodel, module.processes, module.pool)
        # grow the pool
        if module.solpool is not None:
            module.solpool = _update_solution_pool(sol, module.solpool)
        return sol, obj
    # reuse the pool
    return _cache_in_pass(cost, module.optmodel, module.solpool)


def _update_solution_pool(sol, solpool):
    """
    A function to append rows of sol not already in the pool
    """
    sol_uniq = jnp.unique(sol, axis=0)
    # exact-equality via L1 distance (== 0 -> identical row)
    dists = jnp.sum(jnp.abs(sol_uniq[:, None, :] - solpool[None, :, :]), axis=-1)
    is_new = jnp.all(dists != 0, axis=1)
    # host-side gather: the number of new rows is data-dependent
    new_rows = np.asarray(sol_uniq)[np.asarray(is_new)]
    if new_rows.shape[0]:
        solpool = jnp.concatenate([solpool, jnp.asarray(new_rows)], axis=0)
    return solpool


def _cache_in_pass(cost, optmodel, solpool):
    """
    A function to pick the best pool member per instance for cost
    """
    solpool_obj = cost @ solpool.T
    if optmodel.modelSense == EPO.MINIMIZE:
        ind = jnp.argmin(solpool_obj, axis=1)
    else:
        ind = jnp.argmax(solpool_obj, axis=1)
    obj = jnp.take_along_axis(solpool_obj, ind[:, None], axis=1).squeeze(1)
    return solpool[ind], obj


def _full_cost(pred_cost, optmodel):
    """
    A function to lift a predicted cost to the full objective space
    """
    idx = optmodel.c_pred_index
    if idx is None:
        return pred_cost
    prob = optmodel.problem
    full = jnp.broadcast_to(
        jnp.asarray(prob.fixed_cost, dtype=pred_cost.dtype),
        (*pred_cost.shape[:-1], prob.num_vars),
    )
    return full.at[..., jnp.asarray(idx)].add(pred_cost)


def grow_solpool(module, pred_cost):
    """
    A function to re-solve the predicted cost and grow the module's solution pool
    """
    _check_jit_caching(pred_cost, module)
    if module._branch_rng.uniform() <= module.solve_ratio:
        sol, _ = batch_solve(
            jax.lax.stop_gradient(pred_cost), module.optmodel, module.processes, module.pool
        )
        module.solpool = _update_solution_pool(sol, module.solpool)
