#!/usr/bin/env python
"""
Regularized differentiable optimization function (L2 Frank-Wolfe)
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from pyepo import EPO
from pyepo.func.jax.abcmodule import optModule
from pyepo.func.jax.solve import _full_cost, batch_solve


def _sense_sign(optmodel):
    return -1.0 if optmodel.modelSense == EPO.MINIMIZE else 1.0


def _frank_wolfe(theta, optmodel, max_iter, tol):
    """Away-step Frank-Wolfe for argmin_mu 1/2||mu - theta||^2 over conv(S); returns mu."""
    mu, _, _ = _frank_wolfe_active(theta, optmodel, max_iter, tol)
    return mu


def _frank_wolfe_active(theta, optmodel, max_iter, tol):
    """
    Batched away-step Frank-Wolfe with active-set tracking (lax.while_loop);
    returns (mu, vertices, weights).

    Away and drop steps prune the active face, so the iterate reaches the optimal
    face (vanilla Frank-Wolfe stalls) and the returned active set is its true
    support. The buffer width is bounded by the Caratheodory support, independent
    of max_iter. Every instance is solved each step: the away-step gap is not
    monotone, so a per-instance freeze stops short.
    """
    ss = _sense_sign(optmodel)
    b, d = theta.shape
    width = 2 * d + 2
    bidx = jnp.arange(b)
    v0, _ = batch_solve(ss * theta, optmodel)
    vertices = jnp.zeros((b, width, d)).at[:, 0].set(v0)
    weights = jnp.zeros((b, width)).at[:, 0].set(1.0)
    vnorms = jnp.zeros((b, width)).at[:, 0].set(jnp.sum(v0 * v0, axis=-1))

    def cond(state):
        k, _mu, _vt, _w, _vn, unconverged = state
        return jnp.logical_and(k < max_iter, jnp.any(unconverged))

    def body(state):
        k, mu, vt, w, vn, _ = state
        grad = mu - theta
        # Frank-Wolfe vertex and gap, solved for every instance each step
        v, _ = batch_solve(ss * (theta - mu), optmodel)
        gap_fw = jnp.sum(grad * (mu - v), axis=-1)
        # away vertex: active atom maximizing <grad, .>
        scores = jnp.where(w > 0, jnp.einsum("bwv,bv->bw", vt, grad), -jnp.inf)
        away_idx = jnp.argmax(scores, axis=-1)
        v_away = vt[bidx, away_idx]
        alpha_away = w[bidx, away_idx]
        gap_away = jnp.sum(grad * (v_away - mu), axis=-1)
        # zero the step on converged instances per iteration (recoverable, not frozen)
        unconverged = gap_fw >= tol
        active = unconverged.astype(theta.dtype)
        use_fw = gap_fw >= gap_away
        direction = jnp.where(use_fw[:, None], v - mu, mu - v_away)
        gap = jnp.where(use_fw, gap_fw, gap_away)
        gamma_max = jnp.where(use_fw, 1.0, alpha_away / jnp.maximum(1.0 - alpha_away, 1e-12))
        denom = jnp.maximum(jnp.sum(direction * direction, axis=-1), 1e-12)
        gamma = jnp.minimum(jnp.maximum(gap / denom, 0.0), gamma_max) * active
        mu = mu + gamma[:, None] * direction
        # shrink weights: Frank-Wolfe *(1 - gamma), away *(1 + gamma)
        w = w * jnp.where(use_fw, 1.0 - gamma, 1.0 + gamma)[:, None]
        fw = use_fw.astype(theta.dtype)
        gamma_fw = gamma * fw
        gamma_away = gamma * (1.0 - fw)
        # Frank-Wolfe add: dedup against active atoms, else fill a free slot
        vnv = jnp.sum(v * v, axis=-1)
        dist_sq = vn - 2 * jnp.einsum("bwv,bv->bw", vt, v) + vnv[:, None]
        match = (dist_sq < 1e-6) & (w > 0)
        has_match = jnp.any(match, axis=-1)
        match_idx = jnp.argmax(match.astype(theta.dtype), axis=-1)
        free_idx = jnp.argmax((w <= 1e-12).astype(theta.dtype), axis=-1)
        add_new = (~has_match) & use_fw
        w = w.at[bidx, match_idx].add(gamma_fw * (has_match & use_fw).astype(theta.dtype))
        vt = vt.at[bidx, free_idx].set(jnp.where(add_new[:, None], v, vt[bidx, free_idx]))
        vn = vn.at[bidx, free_idx].set(jnp.where(add_new, vnv, vn[bidx, free_idx]))
        w = w.at[bidx, free_idx].set(jnp.where(add_new, gamma_fw, w[bidx, free_idx]))
        # away subtract, then clear FP residue so dropped atoms leave the active set
        w = w.at[bidx, away_idx].add(-gamma_away)
        w = jnp.where(w < 1e-12, 0.0, jnp.maximum(w, 0.0))
        return (k + 1, mu, vt, w, vn, unconverged)

    init = (0, v0, vertices, weights, vnorms, jnp.ones(b, bool))
    _, mu, vertices, weights, _, _ = lax.while_loop(cond, body, init)
    return mu, vertices, weights


class regularizedFrankWolfeOpt(optModule):
    """
    L2-Regularized Frank-Wolfe Optimizer (RFWO) -- differentiable smoothed solver.

    Returns the L2-regularized minimizer over conv(S), solved by batched
    Frank-Wolfe (the only oracle is the standard linear ``optModel`` solve).
    Returns a regularized solution, not a loss -- pair with a task loss, or use
    ``regularizedFrankWolfeFenchelYoung``. The FW loop needs an accurate LMO;
    prefer the callback path with an exact solver (MPAX is approximate here).

    Reference: Dalle et al. (2022) `<https://arxiv.org/abs/2207.13513>`_
    """

    def __init__(
        self,
        optmodel,
        lambd=1.0,
        max_iter=10000,
        tol=1e-6,
        processes=1,
        solve_ratio=1.0,
        dataset=None,
    ):
        """
        Args:
            optmodel: a PyEPO optimization model
            lambd: L2 regularization strength
            max_iter: Frank-Wolfe iteration cap
            tol: per-instance Frank-Wolfe gap tolerance
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of LMO calls solved exactly each step
            dataset: training dataset used to seed the LMO pool when solve_ratio < 1
        """
        super().__init__(optmodel, processes, solve_ratio, dataset=dataset)
        if lambd <= 0:
            raise ValueError("lambda is not positive.")
        self.lambd = float(lambd)
        self.max_iter = max_iter
        self.tol = tol

    def forward(self, pred_cost):
        """
        Forward pass
        """
        # lift to the full objective space
        pred_cost = _full_cost(pred_cost, self.optmodel)
        return _regularized_frank_wolfe_opt(
            pred_cost,
            self.optmodel,
            _sense_sign(self.optmodel) / self.lambd,
            self.max_iter,
            self.tol,
        )


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def _regularized_frank_wolfe_opt(pred_cost, optmodel, scale, max_iter, tol):
    mu, _, _ = _frank_wolfe_active(scale * pred_cost, optmodel, max_iter, tol)
    return mu


def _regularized_frank_wolfe_opt_fwd(pred_cost, optmodel, scale, max_iter, tol):
    mu, vertices, weights = _frank_wolfe_active(scale * pred_cost, optmodel, max_iter, tol)
    return mu, (vertices, weights)


def _regularized_frank_wolfe_opt_bwd(optmodel, scale, max_iter, tol, res, g):
    vertices, weights = res
    # project g onto the affine hull of the active vertices (Gram solve)
    s = (weights > 0).astype(jnp.float32)
    n_active = jnp.clip(s.sum(-1, keepdims=True), 1.0, None)
    v_mean = (vertices * s[..., None]).sum(-2) / n_active
    v_cent = (vertices - v_mean[:, None]) * s[..., None]
    gram = v_cent @ jnp.transpose(v_cent, (0, 2, 1))
    h = v_cent @ g[..., None]
    diag = jnp.diagonal(gram, axis1=-2, axis2=-1)
    ridge = jnp.clip(diag.max(-1, keepdims=True), 1.0, None) * 1e-6
    gram = gram + ridge[..., None] * jnp.eye(gram.shape[-1])
    alpha = jnp.linalg.solve(gram, h)
    grad = jnp.squeeze(jnp.transpose(v_cent, (0, 2, 1)) @ alpha, -1)
    return (scale * grad,)


_regularized_frank_wolfe_opt.defvjp(
    _regularized_frank_wolfe_opt_fwd, _regularized_frank_wolfe_opt_bwd
)


class regularizedFrankWolfeFenchelYoung(optModule):
    """
    L2-Regularized Frank-Wolfe with Fenchel-Young loss (RFYL).

    Pairs the RFWO regularized solver with the Fenchel-Young loss of the L2
    regularizer: a convex scalar comparing the predicted cost to the true
    optimum directly. By Danskin's theorem the gradient is the residual
    ``w - r_sol``, so the backward needs no implicit differentiation.

    Reference: Dalle et al. (2022) `<https://arxiv.org/abs/2207.13513>`_
    """

    def __init__(
        self,
        optmodel,
        lambd=1.0,
        max_iter=10000,
        tol=1e-6,
        processes=1,
        solve_ratio=1.0,
        reduction="mean",
        dataset=None,
    ):
        """
        Args:
            optmodel: a PyEPO optimization model
            lambd: L2 regularization strength
            max_iter: Frank-Wolfe iteration cap
            tol: per-instance Frank-Wolfe gap tolerance
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of LMO calls solved exactly each step
            reduction: reduction applied to the batch loss ("mean", "sum", "none")
            dataset: training dataset used to seed the LMO pool when solve_ratio < 1
        """
        super().__init__(optmodel, processes, solve_ratio, reduction=reduction, dataset=dataset)
        if lambd <= 0:
            raise ValueError("lambda is not positive.")
        self.lambd = float(lambd)
        self.max_iter = max_iter
        self.tol = tol

    def forward(self, pred_cost, true_sol):
        """
        Forward pass
        """
        # lift to the full objective space
        pred_cost = _full_cost(pred_cost, self.optmodel)
        # stop the gradient into the solver
        if self.optmodel.modelSense == EPO.MINIMIZE:
            theta = jax.lax.stop_gradient(-pred_cost / self.lambd)
            r_sol = _frank_wolfe(theta, self.optmodel, self.max_iter, self.tol)
            diff = true_sol - r_sol
        else:
            theta = jax.lax.stop_gradient(pred_cost / self.lambd)
            r_sol = _frank_wolfe(theta, self.optmodel, self.max_iter, self.tol)
            diff = r_sol - true_sol
        omega_w = 0.5 * self.lambd * jnp.sum(true_sol**2, axis=-1)
        omega_r = 0.5 * self.lambd * jnp.sum(r_sol**2, axis=-1)
        loss = (omega_w - omega_r) + jnp.einsum("bi,bi->b", pred_cost, diff)
        return self._reduce(loss)


# acronym aliases
RFWO = regularizedFrankWolfeOpt
RFY = regularizedFrankWolfeFenchelYoung
