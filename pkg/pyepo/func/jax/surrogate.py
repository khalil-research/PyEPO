#!/usr/bin/env python
"""
Surrogate Loss function
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from pyepo import EPO
from pyepo.func.jax.abcmodule import optModule
from pyepo.func.jax.utils import _full_cost, _solve_or_cache
from pyepo.utils import _EPS


class SPOPlus(optModule):
    """
    SPO+ loss: a convex surrogate for the SPO regret of a linear-objective LP.

    SPO+ upper-bounds the SPO regret with a convex function of the predicted
    cost vector and provides an informative subgradient (via Danskin's
    theorem) for end-to-end training. It is the strong default for
    predict-then-optimize when true optimal solutions
    :math:`\\mathbf{w}^*(\\mathbf{c})` are available as supervision.

    Reference: Elmachtoub & Grigas (2022)
    `<https://doi.org/10.1287/mnsc.2020.3922>`_
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1.0, reduction="mean", dataset=None):
        """
        Args:
            optmodel: a PyEPO optimization model
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of instances solved exactly each step
            reduction: reduction applied to the batch loss ("mean", "sum", "none")
            dataset: training dataset used to seed the solution pool when solve_ratio < 1
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)

    def forward(self, pred_cost, true_cost, true_sol, true_obj):
        """
        Forward pass
        """
        # lift to the full objective space
        pred_cost = _full_cost(pred_cost, self.optmodel)
        true_cost = _full_cost(true_cost, self.optmodel)
        loss = _spoplus(pred_cost, true_cost, true_sol, true_obj, self)
        return self._reduce(loss)


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _spoplus(pred_cost, true_cost, true_sol, true_obj, module):
    loss, _ = _spoplus_value_and_grad(pred_cost, true_cost, true_sol, true_obj, module)
    return loss


def _spoplus_value_and_grad(pred_cost, true_cost, true_sol, true_obj, module):
    # solve the perturbed problem
    sol, obj = _solve_or_cache(2.0 * pred_cost - true_cost, module)
    z = jnp.squeeze(true_obj, axis=-1) if true_obj.ndim > 1 else true_obj
    inner = 2.0 * jnp.einsum("bi,bi->b", pred_cost, true_sol)
    # loss and subgradient
    if module.optmodel.modelSense == EPO.MINIMIZE:
        loss = -obj + inner - z
        grad = 2.0 * (true_sol - sol)
    else:
        loss = obj - inner + z
        grad = 2.0 * (sol - true_sol)
    return loss, grad


def _spoplus_fwd(pred_cost, true_cost, true_sol, true_obj, module):
    loss, grad = _spoplus_value_and_grad(pred_cost, true_cost, true_sol, true_obj, module)
    # save subgradient and labels
    return loss, (grad, true_cost, true_sol, true_obj)


def _spoplus_bwd(module, res, g):
    grad, true_cost, true_sol, true_obj = res
    return (
        g[:, None] * grad,
        jnp.zeros_like(true_cost),
        jnp.zeros_like(true_sol),
        jnp.zeros_like(true_obj),
    )


_spoplus.defvjp(_spoplus_fwd, _spoplus_bwd)


class perturbationGradient(optModule):
    """
    Perturbation Gradient (PG): zeroth-order surrogate of the objective-value loss.

    Approximates the directional derivative of the optimal objective along the
    true cost with a finite difference, giving an informative gradient through
    the piecewise-constant solver layer. ``two_sides`` selects backward
    (default) vs central differencing. Needs only the true cost, not the true
    optimal solution.

    Reference: Gupta & Huang (2024) `<https://arxiv.org/abs/2402.03256>`_
    """

    def __init__(
        self,
        optmodel,
        sigma=0.1,
        two_sides=False,
        processes=1,
        solve_ratio=1.0,
        reduction="mean",
        dataset=None,
    ):
        """
        Args:
            optmodel: a PyEPO optimization model
            sigma: finite-difference width
            two_sides: central differencing (True) instead of backward (False)
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of instances solved exactly each step
            reduction: reduction applied to the batch loss ("mean", "sum", "none")
            dataset: training dataset used to seed the solution pool when solve_ratio < 1
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        self.sigma = float(sigma)
        self.two_sides = two_sides

    def forward(self, pred_cost, true_cost):
        """
        Forward pass
        """
        # lift to the full objective space
        pred_cost = _full_cost(pred_cost, self.optmodel)
        true_cost = _full_cost(true_cost, self.optmodel)
        sign = 1.0 if self.optmodel.modelSense == EPO.MINIMIZE else -1.0
        # stop the gradient into the solver
        cp = jax.lax.stop_gradient(pred_cost)
        b = cp.shape[0]
        if self.two_sides:
            # batch +sigma and -sigma into one solve
            combined, _ = _solve_or_cache(
                jnp.concatenate([cp + self.sigma * true_cost, cp - self.sigma * true_cost], axis=0),
                self,
            )
            wp, wm = jax.lax.stop_gradient(combined[:b]), jax.lax.stop_gradient(combined[b:])
            obj_plus = jnp.einsum("bi,bi->b", pred_cost + self.sigma * true_cost, wp)
            obj_minus = jnp.einsum("bi,bi->b", pred_cost - self.sigma * true_cost, wm)
            loss = sign * (obj_plus - obj_minus) / (2 * self.sigma + _EPS)
        else:
            # batch clean and -sigma into one solve
            combined, _ = _solve_or_cache(
                jnp.concatenate([cp, cp - self.sigma * true_cost], axis=0), self
            )
            w, wm = jax.lax.stop_gradient(combined[:b]), jax.lax.stop_gradient(combined[b:])
            obj = jnp.einsum("bi,bi->b", pred_cost, w)
            obj_minus = jnp.einsum("bi,bi->b", pred_cost - self.sigma * true_cost, wm)
            loss = sign * (obj - obj_minus) / (self.sigma + _EPS)
        return self._reduce(loss)


# aliases
smartPredictThenOptimizePlus = SPOPlus
PG = perturbationGradient
