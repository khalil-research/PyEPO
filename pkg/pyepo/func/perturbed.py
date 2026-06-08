#!/usr/bin/env python
"""
Perturbed optimization function
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch.autograd import Function

from pyepo import EPO
from pyepo.func.abcmodule import optModule
from pyepo.func.utils import (
    _mask_pred,
    _torch_generator,
    _update_solution_pool,
    sumGammaDistribution,
)
from pyepo.func.utils import (
    _solve_batch as _solve_batch_2d,
)
from pyepo.utils import _EPS

if TYPE_CHECKING:
    from pyepo.data.dataset import optDataset
    from pyepo.func.abcmodule import Reduction
    from pyepo.model.opt import optModel


class perturbedOpt(optModule):
    """
    Differentiable Perturbed Optimizer (DPO) -- additive-Gaussian variant.

    Estimates the **expected solution**
    :math:`\\mathbb{E}_{\\boldsymbol{\\xi}}[\\mathbf{w}^*(\\hat{\\mathbf{c}} +
    \\sigma\\boldsymbol{\\xi})]` by Monte Carlo averaging over
    ``n_samples`` Gaussian perturbations of the predicted cost vector. The
    smoothed map varies continuously with :math:`\\hat{\\mathbf{c}}` -- small
    perturbations only re-weight the distribution over active vertices --
    giving an informative gradient where the bare LP solver gives zero.

    Returns a solution, not a loss: the user supplies a task loss (MSE
    against :math:`\\mathbf{w}^*(\\mathbf{c})` is the standard choice).
    For sign-sensitive oracles, use ``perturbedOptMul`` instead.

    Reference: Berthet et al. (2020)
    `<https://papers.nips.cc/paper/2020/hash/6bb56208f672af0dd65451f869fedfd9-Abstract.html>`_
    """

    def __init__(
        self,
        optmodel: optModel,
        n_samples: int = 10,
        sigma: float = 1.0,
        processes: int = 1,
        seed: int = 135,
        variance_reduction: bool = True,
        solve_ratio: float = 1.0,
        dataset: optDataset | None = None,
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            n_samples: number of Monte Carlo perturbation samples per instance
            sigma: perturbation amplitude (Gaussian standard deviation)
            processes: number of solver processes (1 = single-core, 0 = all cores)
            seed: random seed for the perturbation generator
            variance_reduction: apply a leave-one-out baseline to the backward estimator
            solve_ratio: fraction of instances solved exactly each step (1.0 = no caching)
            dataset: training dataset used to seed the solution pool when ``solve_ratio < 1``
        """
        super().__init__(optmodel, processes, solve_ratio, dataset=dataset, seed=seed)
        # number of samples
        self.n_samples = n_samples
        # perturbation amplitude
        self.sigma = sigma
        # random state
        self.seed = seed
        # variance reduction
        self.variance_reduction = variance_reduction
        self._gen_cache: dict[str, torch.Generator] = {}

    def forward(self, pred_cost: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # lift costs to the full objective space (no-op without partial prediction)
        pred_cost = self.optmodel._fullCost(pred_cost)
        return cast("torch.Tensor", perturbedOptFunc.apply(pred_cost, self))

    def _perturb(self, cp: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        """Perturbed cost from clean cost (b, d) and noises (b, n, d)."""
        return cp.unsqueeze(1) + self.sigma * noises

    def _grad_scale(self, cp: torch.Tensor) -> torch.Tensor | float:
        """Divisor for the backward estimator."""
        return self.n_samples * self.sigma + _EPS

    def _apply_variance_reduction(self, reward: torch.Tensor) -> torch.Tensor:
        """Apply a leave-one-out baseline to sample rewards."""
        n_samples = reward.shape[1]
        if not self.variance_reduction or n_samples <= 1:
            return reward
        return (reward - reward.mean(dim=1, keepdim=True)) * (n_samples / (n_samples - 1))


class perturbedOptFunc(Function):
    """
    An autograd function for perturbed optimizer
    """

    @staticmethod
    def forward(
        ctx,
        pred_cost: torch.Tensor,
        module: perturbedOpt,
    ) -> torch.Tensor:
        """
        Forward pass for perturbed

        Args:
            pred_cost: a batch of predicted values of the cost
            module: perturbedOpt module

        Returns:
            torch.tensor: solution expectations with perturbation
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach()
        # sample perturbations on-device
        gen = _torch_generator(module._gen_cache, device, module.seed)
        noises = torch.randn(
            (cp.shape[0], module.n_samples, cp.shape[1]),
            device=device,
            dtype=cp.dtype,
            generator=gen,
        )
        # keep known fixed costs unperturbed under partial prediction
        noises = _mask_pred(noises, module.optmodel)
        ptb_c = module._perturb(cp, noises)
        # solve with perturbation
        ptb_sols = _solve_or_cache_3d(ptb_c, module)
        # solution expectation
        e_sol = ptb_sols.mean(dim=1)
        # save solutions
        ctx.save_for_backward(cp, ptb_sols, noises)
        ctx.module = module
        return e_sol

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """
        Backward pass for perturbed
        """
        cp, ptb_sols, noises = ctx.saved_tensors
        reward = torch.einsum("bnd,bd->bn", ptb_sols, grad_output)
        reward = ctx.module._apply_variance_reduction(reward)
        grad = torch.einsum("bnd,bn->bd", noises, reward)
        return grad / ctx.module._grad_scale(cp), None


class perturbedOptMul(perturbedOpt):
    """
    Multiplicative-perturbation variant of ``perturbedOpt`` for sign-sensitive oracles.

    Replaces additive noise with the multiplicative perturbation
    :math:`\\hat{\\mathbf{c}} \\odot \\exp(\\sigma\\boldsymbol{\\xi} -
    \\tfrac{1}{2}\\sigma^2)`, which preserves the sign of each cost entry --
    required when the solver expects, e.g., strictly nonnegative edge costs.
    Predicted costs must already carry the intended nonzero sign; for
    nonnegative problems pair this module with a positive-output prediction
    layer (e.g. ``nn.Softplus()`` plus a small epsilon).

    Reference: Dalle et al. (2022) `<https://arxiv.org/abs/2207.13513>`_
    """

    def _perturb(self, cp: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        return cp.unsqueeze(1) * torch.exp(self.sigma * noises - 0.5 * self.sigma**2)

    def _grad_scale(self, cp: torch.Tensor) -> torch.Tensor:
        denom = self.sigma * cp
        # clamp |denom| >= _EPS, keep sign
        denom_safe = torch.where(
            denom.abs() < _EPS,
            torch.where(denom >= 0, _EPS, -_EPS),
            denom,
        )
        return self.n_samples * denom_safe


class perturbedFenchelYoung(optModule):
    """
    Perturbed Fenchel-Young loss (PFY) -- additive-Gaussian variant.

    Pairs the same Monte-Carlo expected perturbed solution as ``perturbedOpt``
    with the Fenchel-Young loss against the true optimum
    :math:`\\mathbf{w}^*(\\mathbf{c})`, returning a scalar loss directly --
    no user-defined task loss needed. The gradient collapses to the simple
    residual :math:`\\mathbf{w}^*(\\mathbf{c}) - \\mathbb{E}_{\\boldsymbol{\\xi}}
    [\\mathbf{w}^*(\\hat{\\mathbf{c}} + \\sigma\\boldsymbol{\\xi})]`, so no
    explicit Jacobian through the solver is required. For sign-sensitive
    oracles, use ``perturbedFenchelYoungMul``.

    Reference: Berthet et al. (2020)
    `<https://papers.nips.cc/paper/2020/hash/6bb56208f672af0dd65451f869fedfd9-Abstract.html>`_
    """

    def __init__(
        self,
        optmodel: optModel,
        n_samples: int = 10,
        sigma: float = 1.0,
        processes: int = 1,
        seed: int = 135,
        solve_ratio: float = 1.0,
        reduction: Reduction = "mean",
        dataset: optDataset | None = None,
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            n_samples: number of Monte Carlo perturbation samples per instance
            sigma: perturbation amplitude (Gaussian standard deviation)
            processes: number of solver processes (1 = single-core, 0 = all cores)
            seed: random seed for the perturbation generator
            solve_ratio: fraction of instances solved exactly each step (1.0 = no caching)
            reduction: reduction applied to the batch loss (``"mean"``, ``"sum"``, ``"none"``)
            dataset: training dataset used to seed the solution pool when ``solve_ratio < 1``
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset, seed=seed)
        # number of samples
        self.n_samples = n_samples
        # perturbation amplitude
        self.sigma = sigma
        # random state
        self.seed = seed
        self._gen_cache: dict[str, torch.Generator] = {}

    def forward(self, pred_cost: torch.Tensor, true_sol: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # lift costs to the full objective space (no-op without partial prediction)
        pred_cost = self.optmodel._fullCost(pred_cost)
        loss = cast("torch.Tensor", perturbedFenchelYoungFunc.apply(pred_cost, true_sol, self))
        return self._reduce(loss)

    def _perturb(self, cp: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        """Perturbed cost from clean cost (b, d) and noises (b, n, d)."""
        return cp.unsqueeze(1) + self.sigma * noises

    def _calculate_expected_solution(
        self,
        cp: torch.Tensor,
        ptb_c: torch.Tensor,
        ptb_sols: torch.Tensor,
        noises: torch.Tensor,
    ) -> torch.Tensor:
        """First gradient term for perturbed Fenchel-Young."""
        return ptb_sols.mean(dim=1)


class perturbedFenchelYoungFunc(Function):
    """
    An autograd function for Fenchel-Young loss using perturbation techniques.
    """

    @staticmethod
    def forward(
        ctx,
        pred_cost: torch.Tensor,
        true_sol: torch.Tensor,
        module: perturbedFenchelYoung,
    ) -> torch.Tensor:
        """
        Forward pass for perturbed Fenchel-Young loss

        Args:
            pred_cost: a batch of predicted values of the cost
            true_sol: a batch of true optimal solutions
            module: perturbedFenchelYoung module

        Returns:
            torch.tensor: solution expectations with perturbation
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach()
        w = true_sol.detach()
        # sample perturbations on-device
        gen = _torch_generator(module._gen_cache, device, module.seed)
        noises = torch.randn(
            (cp.shape[0], module.n_samples, cp.shape[1]),
            device=device,
            dtype=cp.dtype,
            generator=gen,
        )
        # keep known fixed costs unperturbed under partial prediction
        noises = _mask_pred(noises, module.optmodel)
        ptb_c = module._perturb(cp, noises)
        # solve with perturbation
        ptb_sols = _solve_or_cache_3d(ptb_c, module)
        # solution expectation term in the Fenchel-Young gradient
        e_sol = module._calculate_expected_solution(cp, ptb_c, ptb_sols, noises)
        if module.optmodel.modelSense == EPO.MINIMIZE:
            sign, diff = -1.0, w - e_sol
        else:
            sign, diff = 1.0, e_sol - w
        # Fenchel-Young loss: F(theta) - <theta, true_sol>
        f_theta = torch.einsum("bnd,bnd->bn", ptb_c, ptb_sols).mean(dim=1)
        target_obj = (cp * w).sum(dim=-1)
        loss = sign * (f_theta - target_obj)
        # save solutions
        ctx.save_for_backward(diff)
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """
        Backward pass for perturbed Fenchel-Young loss
        """
        (grad,) = ctx.saved_tensors
        grad_output = torch.unsqueeze(grad_output, dim=-1)
        return grad * grad_output, None, None


class perturbedFenchelYoungMul(perturbedFenchelYoung):
    """
    Multiplicative-perturbation variant of ``perturbedFenchelYoung`` for sign-sensitive oracles.

    Uses the same sign-preserving multiplicative perturbation
    :math:`\\hat{\\mathbf{c}} \\odot \\exp(\\sigma\\boldsymbol{\\xi} -
    \\tfrac{1}{2}\\sigma^2)` as ``perturbedOptMul``. Predicted costs must
    carry the intended nonzero sign; for nonnegative problems pair this
    module with a positive-output prediction layer (e.g. ``nn.Softplus()``
    plus a small epsilon).

    Reference: Dalle et al. (2022) `<https://arxiv.org/abs/2207.13513>`_
    """

    def _perturb(self, cp: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        return cp.unsqueeze(1) * torch.exp(self.sigma * noises - 0.5 * self.sigma**2)

    def _calculate_expected_solution(
        self,
        cp: torch.Tensor,
        ptb_c: torch.Tensor,
        ptb_sols: torch.Tensor,
        noises: torch.Tensor,
    ) -> torch.Tensor:
        factor = torch.exp(self.sigma * noises - 0.5 * self.sigma**2)
        return (ptb_sols * factor).mean(dim=1)


class implicitMLE(optModule):
    """
    Implicit Maximum Likelihood Estimator (I-MLE) via perturb-and-MAP.

    Frames decision-focused learning as imitation: an upstream task gradient
    :math:`\\mathbf{d}` induces a virtual update
    :math:`\\hat{\\mathbf{c}}' = \\hat{\\mathbf{c}} + \\lambda \\mathbf{d}`,
    and the gradient is estimated by a directional finite difference between
    smoothed solutions at :math:`\\hat{\\mathbf{c}}'` and :math:`\\hat{\\mathbf{c}}`,
    sharing the same Sum-of-Gamma noise realization across the two evaluations
    to reduce variance.

    Returns the (perturbation-smoothed) predicted solution; the user supplies
    a task loss (L1 against :math:`\\mathbf{w}^*(\\mathbf{c})` is standard).

    Reference: Niepert, Minervini & Franceschi (2021)
    `<https://proceedings.neurips.cc/paper_files/paper/2021/hash/7a430339c10c642c4b2251756fd1b484-Abstract.html>`_
    """

    def __init__(
        self,
        optmodel: optModel,
        n_samples: int = 10,
        sigma: float = 1.0,
        lambd: float = 10,
        distribution: sumGammaDistribution | None = None,
        two_sides: bool = False,
        processes: int = 1,
        solve_ratio: float = 1.0,
        dataset: optDataset | None = None,
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            n_samples: number of Monte Carlo perturbation samples per instance
            sigma: noise temperature (perturbation amplitude)
            lambd: finite-difference step for the directional gradient estimator
            distribution: noise distribution (defaults to ``sumGammaDistribution(kappa=5)``)
            two_sides: use central differencing instead of backward
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of instances solved exactly each step (1.0 = no caching)
            dataset: training dataset used to seed the solution pool when ``solve_ratio < 1``
        """
        super().__init__(optmodel, processes, solve_ratio, dataset=dataset)
        # number of samples
        self.n_samples = n_samples
        # noise temperature
        self.sigma = sigma
        # smoothing parameter
        if lambd <= 0:
            raise ValueError("lambda is not positive.")
        self.lambd = lambd
        # noise distribution
        if distribution is None:
            distribution = sumGammaDistribution(kappa=5)
        self.distribution = distribution
        # symmetric perturbation
        self.two_sides = two_sides

    def forward(self, pred_cost: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # lift costs to the full objective space (no-op without partial prediction)
        pred_cost = self.optmodel._fullCost(pred_cost)
        return cast("torch.Tensor", implicitMLEFunc.apply(pred_cost, self))


class implicitMLEFunc(Function):
    """
    An autograd function for Implicit Maximum Likelihood Estimator
    """

    @staticmethod
    def forward(
        ctx,
        pred_cost: torch.Tensor,
        module: implicitMLE,
    ) -> torch.Tensor:
        """
        Forward pass for IMLE

        Args:
            pred_cost: a batch of predicted values of the cost
            module: implicitMLE module

        Returns:
            torch.tensor: predicted solutions
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach()
        # sample perturbations; fall back to H2D for custom distributions
        size = (cp.shape[0], module.n_samples, cp.shape[1])
        try:
            noises = module.distribution.sample(size=size, device=device, dtype=cp.dtype)
        except TypeError:
            noises = module.distribution.sample(size=size)
        if isinstance(noises, np.ndarray):
            noises = torch.from_numpy(noises).to(device, dtype=cp.dtype)
        # keep known fixed costs unperturbed under partial prediction
        noises = _mask_pred(noises, module.optmodel)
        ptb_c = cp.unsqueeze(1) + module.sigma * noises
        # solve with perturbation
        ptb_sols = _solve_or_cache_3d(ptb_c, module)
        # solution average
        e_sol = ptb_sols.mean(dim=1)
        # save
        ctx.save_for_backward(pred_cost)
        # add other objects to ctx
        ctx.ptb_c = ptb_c
        ctx.ptb_sols = ptb_sols
        ctx.module = module
        return e_sol

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """
        Backward pass for IMLE
        """
        ptb_c = ctx.ptb_c
        ptb_sols = ctx.ptb_sols
        module = ctx.module
        delta = module.lambd * grad_output.detach().unsqueeze(1)
        if module.two_sides:
            # batch positive and negative perturbations into one solve
            n = module.n_samples
            combined_sols = _solve_or_cache_3d(
                torch.cat([ptb_c + delta, ptb_c - delta], dim=1), module
            )
            ptb_sols_pos, ptb_sols_neg = combined_sols[:, :n], combined_sols[:, n:]
            grad = (ptb_sols_pos - ptb_sols_neg).mean(dim=1) / (2 * module.lambd + _EPS)
        else:
            ptb_sols_pos = _solve_or_cache_3d(ptb_c + delta, module)
            grad = (ptb_sols_pos - ptb_sols).mean(dim=1) / (module.lambd + _EPS)
        return grad, None


class adaptiveImplicitMLE(optModule):
    """
    Adaptive Implicit MLE (AI-MLE): I-MLE with an adaptive interpolation step.

    Replaces I-MLE's fixed finite-difference step :math:`\\lambda` with the
    data-dependent choice :math:`\\lambda_t = \\alpha_t \\cdot \\|\\hat{\\mathbf{c}}\\|
    / \\|\\mathbf{d}\\|`, where the magnitude :math:`\\alpha_t` is tuned online
    from a moving average of gradient sparsity. The rescaling keeps the
    perturbation commensurate with :math:`\\hat{\\mathbf{c}}` and removes the
    need to tune :math:`\\lambda` by hand, while the rest of the forward /
    backward path is identical to ``implicitMLE``.

    Reference: Minervini, Franceschi & Niepert (2023)
    `<https://ojs.aaai.org/index.php/AAAI/article/view/26103>`_
    """

    def __init__(
        self,
        optmodel: optModel,
        n_samples: int = 10,
        sigma: float = 1.0,
        distribution: sumGammaDistribution | None = None,
        two_sides: bool = False,
        processes: int = 1,
        solve_ratio: float = 1.0,
        dataset: optDataset | None = None,
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            n_samples: number of Monte Carlo perturbation samples per instance
            sigma: noise temperature (perturbation amplitude)
            distribution: noise distribution (defaults to ``sumGammaDistribution(kappa=5)``)
            two_sides: use central differencing instead of backward
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of instances solved exactly each step (1.0 = no caching)
            dataset: training dataset used to seed the solution pool when ``solve_ratio < 1``
        """
        super().__init__(optmodel, processes, solve_ratio, dataset=dataset)
        # number of samples
        self.n_samples = n_samples
        # noise temperature
        self.sigma = sigma
        # noise distribution
        if distribution is None:
            distribution = sumGammaDistribution(kappa=5)
        self.distribution = distribution
        # symmetric perturbation
        self.two_sides = two_sides
        # init adaptive params
        self.alpha = 1.0  # adaptive magnitude α
        self.grad_norm_avg = 1  # gradient sparsity estimate
        self.step = 1e-3  # update step for α

    def forward(self, pred_cost: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # lift costs to the full objective space (no-op without partial prediction)
        pred_cost = self.optmodel._fullCost(pred_cost)
        return cast("torch.Tensor", adaptiveImplicitMLEFunc.apply(pred_cost, self))


class adaptiveImplicitMLEFunc(implicitMLEFunc):
    """
    An autograd function for Adaptive Implicit Maximum Likelihood Estimator
    """

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """
        Backward pass for IMLE
        """
        (pred_cost,) = ctx.saved_tensors
        ptb_c = ctx.ptb_c
        ptb_sols = ctx.ptb_sols
        module = ctx.module
        cp = pred_cost.detach()
        dl = grad_output.detach()
        # calculate λ
        dl_norm = torch.norm(dl)
        if dl_norm > 0:
            lambd = module.alpha * torch.norm(cp) / dl_norm
        else:
            lambd = 0.0
        delta = (lambd * dl).unsqueeze(1)
        if module.two_sides:
            # batch positive and negative perturbations into one solve
            n = module.n_samples
            combined_sols = _solve_or_cache_3d(
                torch.cat([ptb_c + delta, ptb_c - delta], dim=1), module
            )
            ptb_sols_pos, ptb_sols_neg = combined_sols[:, :n], combined_sols[:, n:]
            grad = (ptb_sols_pos - ptb_sols_neg).mean(dim=1) / (2 * lambd + _EPS)
        else:
            ptb_sols_pos = _solve_or_cache_3d(ptb_c + delta, module)
            grad = (ptb_sols_pos - ptb_sols).mean(dim=1) / (lambd + _EPS)
        # moving average of the gradient norm
        grad_norm = (grad.abs() > _EPS).float().mean().item()
        module.grad_norm_avg = 0.9 * module.grad_norm_avg + 0.1 * grad_norm
        # update α to target gradient
        if module.grad_norm_avg < 1:
            module.alpha += module.step
        else:
            module.alpha = max(0.0, module.alpha - module.step)
        return grad, None


def _solve_or_cache_3d(ptb_c: torch.Tensor, module: optModule) -> torch.Tensor:
    """
    Solve or use cached solutions for perturbed costs (3D: n_samples × batch × vars).
    Delegates to the shared 2D functions in utils after flattening.
    """
    optmodel = module.optmodel
    processes = module.processes
    pool = module.pool
    solpool = module.solpool
    if module._branch_rng.uniform() <= module.solve_ratio:
        ptb_sols, solpool = _solve_in_pass_3d(ptb_c, optmodel, processes, pool, solpool)
    else:
        # cache branch implies solve_ratio < 1, so __init__ has populated solpool
        assert solpool is not None
        ptb_sols, solpool = _cache_in_pass_3d(ptb_c, optmodel, solpool)
    module.solpool = solpool
    return ptb_sols


def _solve_in_pass_3d(
    ptb_c: torch.Tensor,
    optmodel: optModel,
    processes: int,
    pool,
    solpool: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Solve optimization for perturbed 3D costs and update solution pool.

    Args:
        ptb_c: perturbed costs, shape (batch, n_samples, vars)
        optmodel: optimization model
        processes: number of processors
        pool: process pool
        solpool: solution pool

    Returns:
        tuple: (solutions shape (batch, n_samples, vars), updated solpool)
    """
    ins_num, n_samples, num_vars = ptb_c.shape
    # flatten (batch, n_samples, vars) → (batch * n_samples, vars)
    flat_c = ptb_c.reshape(-1, num_vars)
    # solve using shared 2D function
    flat_sols, _ = _solve_batch_2d(flat_c, optmodel, processes, pool)
    # update pool while flat_sols is still contiguous
    if solpool is not None:
        solpool = _update_solution_pool(flat_sols, solpool)
        if solpool.device != ptb_c.device:
            solpool = solpool.to(ptb_c.device)
    # reshape (batch * n_samples, vars) → (batch, n_samples, vars)
    ptb_sols = flat_sols.reshape(ins_num, n_samples, num_vars)
    return ptb_sols, solpool


def _cache_in_pass_3d(
    ptb_c: torch.Tensor,
    optmodel: optModel,
    solpool: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Use solution pool for perturbed 3D costs (batch × n_samples × vars).
    Unlike the 2D version in utils, this handles the extra sample dimension.
    """
    # move solpool to the correct device
    if solpool.device != ptb_c.device:
        solpool = solpool.to(ptb_c.device)
    # compute objective values: (batch, n_samples, pool_size)
    solpool_obj = torch.einsum("bnd,sd->bns", ptb_c, solpool)
    # best solution in pool
    if optmodel.modelSense == EPO.MINIMIZE:
        best_inds = torch.argmin(solpool_obj, dim=2)
    else:
        best_inds = torch.argmax(solpool_obj, dim=2)
    ptb_sols = solpool[best_inds]
    return ptb_sols, solpool


# acronym aliases
DPO = perturbedOpt
DPOMul = perturbedOptMul
PFY = perturbedFenchelYoung
PFYMul = perturbedFenchelYoungMul
IMLE = implicitMLE
AIMLE = adaptiveImplicitMLE
