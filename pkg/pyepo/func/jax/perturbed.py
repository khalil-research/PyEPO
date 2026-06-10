#!/usr/bin/env python
"""
Perturbed optimization function
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from pyepo import EPO
from pyepo.func.jax.abcmodule import optModule
from pyepo.func.jax.utils import _full_cost, _mask_pred, _solve_or_cache, _sum_gamma_sample
from pyepo.utils import _EPS


class perturbedOpt(optModule):
    """
    Differentiable Perturbed Optimizer (DPO) -- additive-Gaussian variant.

    Estimates the expected solution
    :math:`\\mathbb{E}_{\\boldsymbol{\\xi}}[\\mathbf{w}^*(\\hat{\\mathbf{c}} +
    \\sigma\\boldsymbol{\\xi})]` by Monte Carlo averaging, giving an
    informative gradient where the bare solver gives zero. Returns a solution;
    pair with a task loss.

    Reference: Berthet et al. (2020)
    `<https://papers.nips.cc/paper/2020/hash/6bb56208f672af0dd65451f869fedfd9-Abstract.html>`_
    """

    _multiplicative = False

    def __init__(
        self,
        optmodel,
        n_samples=10,
        sigma=1.0,
        processes=1,
        seed=135,
        variance_reduction=True,
        solve_ratio=1.0,
        dataset=None,
    ):
        """
        Args:
            optmodel: a PyEPO optimization model
            n_samples: number of Monte Carlo perturbation samples per instance
            sigma: perturbation amplitude (Gaussian standard deviation)
            processes: number of solver processes (1 = single-core, 0 = all cores)
            seed: random seed for the perturbation generator
            variance_reduction: apply a leave-one-out baseline in the backward estimator
            solve_ratio: fraction of instances solved exactly each step
            dataset: training dataset used to seed the solution pool when solve_ratio < 1
        """
        super().__init__(optmodel, processes, solve_ratio, dataset=dataset, seed=seed)
        self.n_samples = n_samples
        self.sigma = float(sigma)
        self.variance_reduction = variance_reduction
        self._key = jax.random.PRNGKey(seed)

    def forward(self, pred_cost, key=None):
        """
        Forward pass
        """
        # lift to the full objective space
        pred_cost = _full_cost(pred_cost, self.optmodel)
        # explicit key -> jittable; None -> eager, advance the instance key
        if key is None:
            self._key, key = jax.random.split(self._key)
        noises = jax.random.normal(
            key, (pred_cost.shape[0], self.n_samples, pred_cost.shape[1]), dtype=pred_cost.dtype
        )
        # keep fixed costs unperturbed
        noises = _mask_pred(noises, self.optmodel)
        return _perturbed_opt(
            pred_cost, noises, self, self.sigma, self.variance_reduction, self._multiplicative
        )


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def _perturbed_opt(pred_cost, noises, module, sigma, variance_reduction, multiplicative):
    e_sol, _ = _perturbed_opt_core(pred_cost, noises, module, sigma, multiplicative)
    return e_sol


def _perturbed_opt_core(pred_cost, noises, module, sigma, multiplicative):
    ptb_c = _perturb(pred_cost, noises, sigma, multiplicative)
    ptb_sols = _solve_or_cache_3d(ptb_c, module)
    return ptb_sols.mean(axis=1), ptb_sols


def _perturbed_opt_fwd(pred_cost, noises, module, sigma, variance_reduction, multiplicative):
    e_sol, ptb_sols = _perturbed_opt_core(pred_cost, noises, module, sigma, multiplicative)
    return e_sol, (pred_cost, ptb_sols, noises)


def _perturbed_opt_bwd(module, sigma, variance_reduction, multiplicative, res, g):
    pred_cost, ptb_sols, noises = res
    n = noises.shape[1]
    # reward-weighted estimator
    reward = jnp.einsum("bnd,bd->bn", ptb_sols, g)
    if variance_reduction and n > 1:
        reward = (reward - reward.mean(axis=1, keepdims=True)) * (n / (n - 1))
    if multiplicative:
        denom = sigma * pred_cost
        denom_safe = jnp.where(jnp.abs(denom) < _EPS, jnp.where(denom >= 0, _EPS, -_EPS), denom)
        grad = jnp.einsum("bnd,bn->bd", noises, reward) / (n * denom_safe)
    else:
        grad = jnp.einsum("bnd,bn->bd", noises, reward) / (n * sigma + _EPS)
    return (grad, jnp.zeros_like(noises))


_perturbed_opt.defvjp(_perturbed_opt_fwd, _perturbed_opt_bwd)


class perturbedOptMul(perturbedOpt):
    """
    Differentiable Perturbed Optimizer (DPO) -- multiplicative log-normal variant.

    As :class:`perturbedOpt`, but perturbs the cost multiplicatively with
    log-normal noise :math:`\\exp(\\sigma\\boldsymbol{\\xi} - \\sigma^2/2)`.

    Reference: Dalle et al. (2022) `<https://arxiv.org/abs/2207.13513>`_
    """

    _multiplicative = True


class perturbedFenchelYoung(optModule):
    """
    Perturbed Fenchel-Young loss (PFYL) -- additive-Gaussian variant.

    Pairs a Monte-Carlo expected perturbed solution with the Fenchel-Young
    loss against the true optimum, returning a scalar loss whose gradient is
    the residual :math:`\\mathbf{w}^*(\\mathbf{c}) -
    \\mathbb{E}_{\\boldsymbol{\\xi}}[\\mathbf{w}^*(\\hat{\\mathbf{c}} +
    \\sigma\\boldsymbol{\\xi})]`.

    Reference: Berthet et al. (2020)
    `<https://papers.nips.cc/paper/2020/hash/6bb56208f672af0dd65451f869fedfd9-Abstract.html>`_
    """

    _multiplicative = False

    def __init__(
        self,
        optmodel,
        n_samples=10,
        sigma=1.0,
        processes=1,
        seed=135,
        solve_ratio=1.0,
        reduction="mean",
        dataset=None,
    ):
        """
        Args:
            optmodel: a PyEPO optimization model
            n_samples: number of Monte Carlo perturbation samples per instance
            sigma: perturbation amplitude (Gaussian standard deviation)
            processes: number of solver processes (1 = single-core, 0 = all cores)
            seed: random seed for the perturbation generator
            solve_ratio: fraction of instances solved exactly each step
            reduction: reduction applied to the batch loss ("mean", "sum", "none")
            dataset: training dataset used to seed the solution pool when solve_ratio < 1
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset, seed=seed)
        self.n_samples = n_samples
        self.sigma = float(sigma)
        self._key = jax.random.PRNGKey(seed)

    def forward(self, pred_cost, true_sol, key=None):
        """
        Forward pass
        """
        # lift to the full objective space
        pred_cost = _full_cost(pred_cost, self.optmodel)
        # explicit key -> jittable; None -> eager, advance the instance key
        if key is None:
            self._key, key = jax.random.split(self._key)
        noises = jax.random.normal(
            key, (pred_cost.shape[0], self.n_samples, pred_cost.shape[1]), dtype=pred_cost.dtype
        )
        # keep fixed costs unperturbed
        noises = _mask_pred(noises, self.optmodel)
        loss = _perturbed_fenchel_young(
            pred_cost, true_sol, noises, self, self.sigma, self._multiplicative
        )
        return self._reduce(loss)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _perturbed_fenchel_young(pred_cost, true_sol, noises, module, sigma, multiplicative):
    loss, _ = _perturbed_fenchel_young_value_and_grad(
        pred_cost, true_sol, noises, module, sigma, multiplicative
    )
    return loss


def _perturbed_fenchel_young_value_and_grad(
    pred_cost, true_sol, noises, module, sigma, multiplicative
):
    # perturb and solve
    ptb_c = _perturb(pred_cost, noises, sigma, multiplicative)
    ptb_sols = _solve_or_cache_3d(ptb_c, module)
    # expected solution
    if multiplicative:
        factor = jnp.exp(sigma * noises - 0.5 * sigma**2)
        e_sol = (ptb_sols * factor).mean(axis=1)
    else:
        e_sol = ptb_sols.mean(axis=1)
    # fenchel-young value and residual gradient
    f_theta = jnp.einsum("bnd,bnd->bn", ptb_c, ptb_sols).mean(axis=1)
    target_obj = jnp.einsum("bd,bd->b", pred_cost, true_sol)
    if module.optmodel.modelSense == EPO.MINIMIZE:
        loss = -(f_theta - target_obj)
        diff = true_sol - e_sol
    else:
        loss = f_theta - target_obj
        diff = e_sol - true_sol
    return loss, diff


def _perturbed_fenchel_young_fwd(pred_cost, true_sol, noises, module, sigma, multiplicative):
    loss, diff = _perturbed_fenchel_young_value_and_grad(
        pred_cost, true_sol, noises, module, sigma, multiplicative
    )
    return loss, (diff, true_sol, noises)


def _perturbed_fenchel_young_bwd(module, sigma, multiplicative, res, g):
    diff, true_sol, noises = res
    return (g[:, None] * diff, jnp.zeros_like(true_sol), jnp.zeros_like(noises))


_perturbed_fenchel_young.defvjp(_perturbed_fenchel_young_fwd, _perturbed_fenchel_young_bwd)


class perturbedFenchelYoungMul(perturbedFenchelYoung):
    """
    Perturbed Fenchel-Young loss (PFYL) -- multiplicative log-normal variant.

    As :class:`perturbedFenchelYoung`, but perturbs the cost multiplicatively
    with log-normal noise :math:`\\exp(\\sigma\\boldsymbol{\\xi} - \\sigma^2/2)`.

    Reference: Dalle et al. (2022) `<https://arxiv.org/abs/2207.13513>`_
    """

    _multiplicative = True


class implicitMLE(optModule):
    """
    Implicit Maximum Likelihood Estimator (I-MLE) via perturb-and-MAP.

    Frames decision-focused learning as imitation: an upstream gradient
    induces a virtual update :math:`\\hat{\\mathbf{c}} + \\lambda \\mathbf{d}`,
    and the gradient is a directional finite difference between smoothed
    solutions at the updated and original costs, sharing one Sum-of-Gamma noise
    realization across both.

    Reference: Niepert, Minervini & Franceschi (2021)
    `<https://proceedings.neurips.cc/paper_files/paper/2021/hash/7a430339c10c642c4b2251756fd1b484-Abstract.html>`_
    """

    def __init__(
        self,
        optmodel,
        n_samples=10,
        sigma=1.0,
        lambd=10,
        kappa=5,
        n_iterations=10,
        two_sides=False,
        seed=135,
        processes=1,
        solve_ratio=1.0,
        dataset=None,
    ):
        """
        Args:
            optmodel: a PyEPO optimization model
            n_samples: number of Monte Carlo perturbation samples per instance
            sigma: noise temperature (perturbation amplitude)
            lambd: finite-difference step for the directional gradient
            kappa: Sum-of-Gamma shape parameter
            n_iterations: Sum-of-Gamma summation length
            two_sides: use central differencing instead of backward
            seed: random seed for the perturbation generator
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of instances solved exactly each step
            dataset: training dataset used to seed the solution pool when solve_ratio < 1
        """
        super().__init__(optmodel, processes, solve_ratio, dataset=dataset, seed=seed)
        if lambd <= 0:
            raise ValueError("lambda is not positive.")
        self.n_samples = n_samples
        self.sigma = float(sigma)
        self.lambd = float(lambd)
        self.kappa = float(kappa)
        self.n_iterations = n_iterations
        self.two_sides = two_sides
        self._key = jax.random.PRNGKey(seed)

    def forward(self, pred_cost, key=None):
        """
        Forward pass
        """
        # lift to the full objective space
        pred_cost = _full_cost(pred_cost, self.optmodel)
        # explicit key -> jittable; None -> eager, advance the instance key
        if key is None:
            self._key, key = jax.random.split(self._key)
        noises = _sum_gamma_sample(
            key,
            self.kappa,
            self.n_iterations,
            (pred_cost.shape[0], self.n_samples, pred_cost.shape[1]),
        )
        # keep fixed costs unperturbed
        noises = _mask_pred(noises, self.optmodel)
        return _implicit_mle(pred_cost, noises, self, self.sigma, self.lambd, self.two_sides)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def _implicit_mle(pred_cost, noises, module, sigma, lambd, two_sides):
    e_sol, _ = _implicit_mle_core(pred_cost, noises, module, sigma)
    return e_sol


def _implicit_mle_core(pred_cost, noises, module, sigma):
    ptb_c = pred_cost[:, None, :] + sigma * noises
    ptb_sols = _solve_or_cache_3d(ptb_c, module)
    return ptb_sols.mean(axis=1), (ptb_c, ptb_sols)


def _implicit_mle_fwd(pred_cost, noises, module, sigma, lambd, two_sides):
    e_sol, (ptb_c, ptb_sols) = _implicit_mle_core(pred_cost, noises, module, sigma)
    return e_sol, (ptb_c, ptb_sols, noises)


def _implicit_mle_bwd(module, sigma, lambd, two_sides, res, g):
    ptb_c, ptb_sols, noises = res
    # finite-difference re-solve along the upstream gradient
    delta = lambd * g[:, None, :]
    if two_sides:
        # batch +delta and -delta into one solve
        n = noises.shape[1]
        both = _solve_or_cache_3d(jnp.concatenate([ptb_c + delta, ptb_c - delta], axis=1), module)
        grad = (both[:, :n] - both[:, n:]).mean(axis=1) / (2 * lambd + _EPS)
    else:
        grad = (_solve_or_cache_3d(ptb_c + delta, module) - ptb_sols).mean(axis=1) / (lambd + _EPS)
    return (grad, jnp.zeros_like(noises))


_implicit_mle.defvjp(_implicit_mle_fwd, _implicit_mle_bwd)


class adaptiveImplicitMLE(optModule):
    """
    Adaptive Implicit MLE (AI-MLE): I-MLE with an online-tuned interpolation step.

    Replaces I-MLE's fixed lambda with :math:`\\lambda_t = \\alpha_t
    \\|\\hat{\\mathbf{c}}\\| / \\|\\mathbf{d}\\|`, where :math:`\\alpha_t` is
    adapted online from a moving average of the gradient sparsity. Eager-only:
    the alpha update is a concrete side effect in the backward, so this loss is
    not ``jax.jit``-able.

    Reference: Minervini, Franceschi & Niepert (2023)
    `<https://ojs.aaai.org/index.php/AAAI/article/view/26103>`_
    """

    def __init__(
        self,
        optmodel,
        n_samples=10,
        sigma=1.0,
        kappa=5,
        n_iterations=10,
        two_sides=False,
        seed=135,
        processes=1,
        solve_ratio=1.0,
        dataset=None,
    ):
        """
        Args:
            optmodel: a PyEPO optimization model
            n_samples: number of Monte Carlo perturbation samples per instance
            sigma: noise temperature (perturbation amplitude)
            kappa: Sum-of-Gamma shape parameter
            n_iterations: Sum-of-Gamma summation length
            two_sides: use central differencing instead of backward
            seed: random seed for the perturbation generator
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of instances solved exactly each step
            dataset: training dataset used to seed the solution pool when solve_ratio < 1
        """
        super().__init__(optmodel, processes, solve_ratio, dataset=dataset, seed=seed)
        self.n_samples = n_samples
        self.sigma = float(sigma)
        self.kappa = float(kappa)
        self.n_iterations = n_iterations
        self.two_sides = two_sides
        self._key = jax.random.PRNGKey(seed)
        # adaptive state (mutated in the backward)
        self.alpha = 1.0
        self.grad_norm_avg = 1.0
        self.step = 1e-3

    def forward(self, pred_cost):
        """
        Forward pass
        """
        # lift to the full objective space
        pred_cost = _full_cost(pred_cost, self.optmodel)
        self._key, sub = jax.random.split(self._key)
        noises = _sum_gamma_sample(
            sub,
            self.kappa,
            self.n_iterations,
            (pred_cost.shape[0], self.n_samples, pred_cost.shape[1]),
        )
        # keep fixed costs unperturbed
        noises = _mask_pred(noises, self.optmodel)
        return _adaptive_implicit_mle(pred_cost, noises, self)


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _adaptive_implicit_mle(pred_cost, noises, module):
    e_sol, _ = _implicit_mle_core(pred_cost, noises, module, module.sigma)
    return e_sol


def _adaptive_implicit_mle_fwd(pred_cost, noises, module):
    e_sol, (ptb_c, ptb_sols) = _implicit_mle_core(pred_cost, noises, module, module.sigma)
    return e_sol, (pred_cost, ptb_c, ptb_sols, noises)


def _adaptive_implicit_mle_bwd(module, res, g):
    pred_cost, ptb_c, ptb_sols, noises = res
    # adaptive step from the upstream cotangent
    dl_norm = jnp.linalg.norm(g)
    lambd = (
        float(module.alpha * jnp.linalg.norm(pred_cost) / dl_norm) if float(dl_norm) > 0 else 0.0
    )
    delta = lambd * g[:, None, :]
    if module.two_sides:
        # batch +delta and -delta into one solve
        n = noises.shape[1]
        both = _solve_or_cache_3d(jnp.concatenate([ptb_c + delta, ptb_c - delta], axis=1), module)
        grad = (both[:, :n] - both[:, n:]).mean(axis=1) / (2 * lambd + _EPS)
    else:
        grad = (_solve_or_cache_3d(ptb_c + delta, module) - ptb_sols).mean(axis=1) / (lambd + _EPS)
    # online alpha update
    grad_norm = float((jnp.abs(grad) > _EPS).mean())
    module.grad_norm_avg = 0.9 * module.grad_norm_avg + 0.1 * grad_norm
    module.alpha = (
        module.alpha + module.step
        if module.grad_norm_avg < 1
        else max(0.0, module.alpha - module.step)
    )
    return (grad, jnp.zeros_like(noises))


_adaptive_implicit_mle.defvjp(_adaptive_implicit_mle_fwd, _adaptive_implicit_mle_bwd)


def _solve_or_cache_3d(ptb_c, module):
    """
    A function to solve perturbed 3D costs (batch, n_samples, vars) with caching
    """
    b, n, d = ptb_c.shape
    sol, _ = _solve_or_cache(ptb_c.reshape(-1, d), module)
    return sol.reshape(b, n, d)


def _perturb(pred_cost, noises, sigma, multiplicative):
    """
    A function to perturb the cost additively or multiplicatively
    """
    if multiplicative:
        return pred_cost[:, None, :] * jnp.exp(sigma * noises - 0.5 * sigma**2)
    return pred_cost[:, None, :] + sigma * noises


# acronym aliases
DPO = perturbedOpt
DPOMul = perturbedOptMul
PFY = perturbedFenchelYoung
PFYMul = perturbedFenchelYoungMul
IMLE = implicitMLE
AIMLE = adaptiveImplicitMLE
