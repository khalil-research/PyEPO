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
from pyepo.func.jax.solve import solve_or_cache


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
            processes: number of solver processes (1 = single-core)
            solve_ratio: fraction of instances solved exactly each step
            reduction: reduction applied to the batch loss ("mean", "sum", "none")
            dataset: training dataset used to seed the solution pool when solve_ratio < 1
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)

    def forward(self, pred_cost, true_cost, true_sol, true_obj):
        """
        Forward pass
        """
        loss = _spoplus(pred_cost, true_cost, true_sol, true_obj, self)
        return self._reduce(loss)


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _spoplus(pred_cost, true_cost, true_sol, true_obj, module):
    loss, _ = _spoplus_value_and_grad(pred_cost, true_cost, true_sol, true_obj, module)
    return loss


def _spoplus_value_and_grad(pred_cost, true_cost, true_sol, true_obj, module):
    # solve the perturbed problem
    sol, obj = solve_or_cache(2.0 * pred_cost - true_cost, module)
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


# aliases
smartPredictThenOptimizePlus = SPOPlus
