#!/usr/bin/env python
"""
Differentiable Black-box optimization function
"""

from __future__ import annotations

from functools import partial

import jax

from pyepo import EPO
from pyepo.func.jax.abcmodule import optModule
from pyepo.func.jax.solve import solve_or_cache
from pyepo.utils import _EPS


class negativeIdentity(optModule):
    """
    Negative Identity Backpropagation (NID) -- hyperparameter-free DBB.

    Treats the solver Jacobian as a signed identity:
    :math:`\\partial \\mathbf{w}^* / \\partial \\hat{\\mathbf{c}} \\approx
    -\\mathbf{I}` for minimization (and :math:`+\\mathbf{I}` for maximization),
    a straight-through gradient estimator needing no extra solve.

    Reference: Sahoo et al. (2022) `<https://arxiv.org/abs/2205.15213>`_
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1.0, dataset=None):
        """
        Args:
            optmodel: a PyEPO optimization model
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of instances solved exactly each step
            dataset: training dataset used to seed the solution pool when solve_ratio < 1
        """
        super().__init__(optmodel, processes, solve_ratio, dataset=dataset)

    def forward(self, pred_cost):
        """
        Forward pass
        """
        return _negative_identity(pred_cost, self)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _negative_identity(pred_cost, module):
    sol, _ = solve_or_cache(pred_cost, module)
    return sol


def _negative_identity_fwd(pred_cost, module):
    sol, _ = solve_or_cache(pred_cost, module)
    return sol, None


def _negative_identity_bwd(module, _res, g):
    # negative identity gradient
    if module.optmodel.modelSense == EPO.MINIMIZE:
        return (-g,)
    return (g,)


_negative_identity.defvjp(_negative_identity_fwd, _negative_identity_bwd)


class blackboxOpt(optModule):
    """
    Differentiable Black-Box Optimizer (DBB) -- gradient via solution interpolation.

    Replaces the solver's zero gradient with an interpolation estimate: for an
    upstream gradient :math:`\\mathbf{d}`, the vector-Jacobian product is
    :math:`(\\mathbf{w}^*(\\hat{\\mathbf{c}} + \\lambda \\mathbf{d}) -
    \\mathbf{w}^*(\\hat{\\mathbf{c}})) / \\lambda`. Larger ``lambd`` smooths
    more (recommended 10-20).

    Reference: Vlastelica et al. (2019) `<https://arxiv.org/abs/1912.02175>`_
    """

    def __init__(self, optmodel, lambd=10, processes=1, solve_ratio=1.0, dataset=None):
        """
        Args:
            optmodel: a PyEPO optimization model
            lambd: interpolation smoothing strength (recommended 10-20)
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of instances solved exactly each step
            dataset: training dataset used to seed the solution pool when solve_ratio < 1
        """
        super().__init__(optmodel, processes, solve_ratio, dataset=dataset)
        if lambd <= 0:
            raise ValueError("lambda is not positive.")
        self.lambd = float(lambd)

    def forward(self, pred_cost):
        """
        Forward pass
        """
        return _blackbox_opt(pred_cost, self, self.lambd)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _blackbox_opt(pred_cost, module, lambd):
    sol, _ = solve_or_cache(pred_cost, module)
    return sol


def _blackbox_opt_fwd(pred_cost, module, lambd):
    sol, _ = solve_or_cache(pred_cost, module)
    return sol, (pred_cost, sol)


def _blackbox_opt_bwd(module, lambd, res, g):
    pred_cost, wp = res
    # perturbed solve in the backward pass
    sol, _ = solve_or_cache(pred_cost + lambd * g, module)
    # interpolation gradient
    grad = (sol - wp) / (lambd + _EPS)
    return (grad,)


_blackbox_opt.defvjp(_blackbox_opt_fwd, _blackbox_opt_bwd)


# acronym aliases
DBB = blackboxOpt
NID = negativeIdentity
