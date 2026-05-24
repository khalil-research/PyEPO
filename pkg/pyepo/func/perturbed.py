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
    _solve_batch as _solve_batch_2d,
)
from pyepo.func.utils import (
    _torch_generator,
    _update_solution_pool,
    sumGammaDistribution,
)
from pyepo.utils import _EPS

if TYPE_CHECKING:
    from pyepo.data.dataset import optDataset
    from pyepo.func.abcmodule import Reduction
    from pyepo.model.opt import optModel


class perturbedOpt(optModule):
    """
    A differentiable perturbed optimizer that estimates the expected solution
    by solving randomly perturbed cost vectors.

    For the perturbed optimizer, the cost vector is predicted from contextual
    data and perturbed with Gaussian noise.

    The custom backward pass provides a Monte Carlo gradient estimator for
    stochastic gradient descent.

    Reference: <https://papers.nips.cc/paper/2020/hash/6bb56208f672af0dd65451f869fedfd9-Abstract.html>
    """

    def __init__(
        self,
        optmodel: optModel,
        n_samples: int = 10,
        sigma: float = 1.0,
        processes: int = 1,
        seed: int = 135,
        solve_ratio: float = 1.0,
        dataset: optDataset | None = None,
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            n_samples: number of Monte Carlo samples
            sigma: the amplitude of the perturbation
            processes: number of processors, 1 for single-core, 0 for all of cores
            seed: random seed
            solve_ratio: the ratio of new solutions computed during training
            dataset: the training data
        """
        super().__init__(optmodel, processes, solve_ratio, dataset=dataset, seed=seed)
        # number of samples
        self.n_samples = n_samples
        # perturbation amplitude
        self.sigma = sigma
        # random state
        self.seed = seed
        self._gen_cache: dict[str, torch.Generator] = {}

    def forward(self, pred_cost: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        return cast("torch.Tensor", perturbedOptFunc.apply(pred_cost, self))

    def _perturb(self, cp: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        """Perturbed cost from clean cost and noises."""
        return cp + self.sigma * noises

    def _grad_scale(self, cp: torch.Tensor) -> torch.Tensor | float:
        """Divisor for the backward estimator."""
        return self.n_samples * self.sigma + _EPS


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
            (module.n_samples, *cp.shape), device=device, dtype=torch.float32, generator=gen,
        )
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
        grad = torch.einsum("nbd,bn->bd", noises, torch.einsum("bnd,bd->bn", ptb_sols, grad_output))
        return grad / ctx.module._grad_scale(cp), None


class perturbedOptMul(perturbedOpt):
    """
    Multiplicative-perturbation variant of perturbedOpt.

    The perturbation ``cp * exp(sigma * noise - sigma**2 / 2)`` preserves the
    sign of each cost entry, which is useful for sign-sensitive oracles.
    This estimator assumes predicted costs already have the intended nonzero
    sign. For nonnegative-cost problems, use a positive-output prediction layer,
    such as Softplus plus a small epsilon.

    Reference: <https://arxiv.org/abs/2207.13513>
    """

    def _perturb(self, cp: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        return cp * torch.exp(self.sigma * noises - 0.5 * self.sigma**2)

    def _grad_scale(self, cp: torch.Tensor) -> torch.Tensor:
        denom = self.sigma * cp
        sign = torch.where(denom >= 0, torch.ones_like(denom), -torch.ones_like(denom))
        denom_safe = torch.where(denom.abs() < _EPS, sign * _EPS, denom)
        return self.n_samples * denom_safe


class perturbedFenchelYoung(optModule):
    """
    A perturbed Fenchel-Young loss using Monte Carlo perturbations.

    The cost vector is predicted from contextual data and perturbed with
    Gaussian noise.

    This loss directly compares the expected perturbed solution with the true
    optimal solution, avoiding the extra task loss needed by perturbedOpt.

    Reference: <https://papers.nips.cc/paper/2020/hash/6bb56208f672af0dd65451f869fedfd9-Abstract.html>
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
            n_samples: number of Monte Carlo samples
            sigma: the amplitude of the perturbation
            processes: number of processors, 1 for single-core, 0 for all of cores
            seed: random seed
            solve_ratio: the ratio of new solutions computed during training
            reduction: the reduction to apply to the output
            dataset: the training data
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
        loss = cast("torch.Tensor", perturbedFenchelYoungFunc.apply(pred_cost, true_sol, self))
        return self._reduce(loss)

    def _perturb(self, cp: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        """Perturbed cost from clean cost and noises."""
        return cp + self.sigma * noises

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
            (module.n_samples, *cp.shape), device=device, dtype=torch.float32, generator=gen,
        )
        ptb_c = module._perturb(cp, noises)
        # solve with perturbation
        ptb_sols = _solve_or_cache_3d(ptb_c, module)
        # solution expectation term in the Fenchel-Young gradient
        e_sol = module._calculate_expected_solution(cp, ptb_c, ptb_sols, noises)
        if module.optmodel.modelSense == EPO.MINIMIZE:
            sign = -1.0
        elif module.optmodel.modelSense == EPO.MAXIMIZE:
            sign = 1.0
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        # difference
        if module.optmodel.modelSense == EPO.MINIMIZE:
            diff = w - e_sol
        elif module.optmodel.modelSense == EPO.MAXIMIZE:
            diff = e_sol - w
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        # Fenchel-Young loss: F(theta) - <theta, true_sol>
        ptb_c_b = ptb_c.permute(1, 0, 2)
        f_theta = torch.einsum("bnd,bnd->bn", sign * ptb_c_b, ptb_sols).mean(dim=1)
        target_obj = torch.einsum("bd,bd->b", sign * cp, w)
        loss = f_theta - target_obj
        # save solutions
        ctx.save_for_backward(diff)
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """
        Backward pass for perturbed Fenchel-Young loss
        """
        grad, = ctx.saved_tensors
        grad_output = torch.unsqueeze(grad_output, dim=-1)
        return grad * grad_output, None, None


class perturbedFenchelYoungMul(perturbedFenchelYoung):
    """
    Multiplicative-perturbation variant of perturbedFenchelYoung.

    This variant preserves the sign of each predicted cost entry. It assumes
    predicted costs already have the intended nonzero sign; for nonnegative-cost
    problems, use a positive-output prediction layer.

    Reference: <https://arxiv.org/abs/2207.13513>
    """

    def _perturb(self, cp: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        return cp * torch.exp(self.sigma * noises - 0.5 * self.sigma**2)

    def _calculate_expected_solution(
        self,
        cp: torch.Tensor,
        ptb_c: torch.Tensor,
        ptb_sols: torch.Tensor,
        noises: torch.Tensor,
    ) -> torch.Tensor:
        factor = torch.exp(self.sigma * noises - 0.5 * self.sigma**2)
        return (ptb_sols * factor.permute(1, 0, 2)).mean(dim=1)


class implicitMLE(optModule):
    """
    An autograd module for Implicit Maximum Likelihood Estimator, which yields
    an optimal solution in a constrained exponential family distribution via
    Perturb-and-MAP.

    For I-MLE, it works as black-box combinatorial solvers, in which constraints
    are known and fixed, but the cost vector needs to be predicted from
    contextual data.

    The I-MLE approximates the gradient of the optimizer smoothly. Thus, it allows us to
    design an algorithm based on stochastic gradient descent.

    Reference: <https://proceedings.neurips.cc/paper_files/paper/2021/hash/7a430339c10c642c4b2251756fd1b484-Abstract.html>
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
            n_samples: number of Monte Carlo samples
            sigma: noise temperature for the input distribution
            lambd: a hyperparameter for differentiable black-box to control interpolation degree
            distribution: noise distribution
            two_sides: approximate gradient by two-sided perturbation or not
            processes: number of processors, 1 for single-core, 0 for all of cores
            solve_ratio: the ratio of new solutions computed during training
            dataset: the training data
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
        size = (module.n_samples, *cp.shape)
        try:
            noises = module.distribution.sample(size=size, device=device, dtype=torch.float32)
        except TypeError:
            noises = module.distribution.sample(size=size)
        if isinstance(noises, np.ndarray):
            noises = torch.from_numpy(noises).to(device, dtype=torch.float32)
        ptb_c = cp + module.sigma * noises
        # solve with perturbation
        ptb_sols = _solve_or_cache_3d(ptb_c, module)
        # solution average
        e_sol = ptb_sols.mean(dim=1)
        # save
        ctx.save_for_backward(pred_cost)
        # add other objects to ctx
        ctx.noises = noises
        ctx.ptb_sols = ptb_sols
        ctx.module = module
        return e_sol

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """
        Backward pass for IMLE
        """
        pred_cost, = ctx.saved_tensors
        noises = ctx.noises
        ptb_sols = ctx.ptb_sols
        module = ctx.module
        # convert tensor
        cp = pred_cost.detach()
        dl = grad_output.detach()
        # positive perturbed costs
        ptb_cp_pos = cp + module.lambd * dl + module.sigma * noises
        # solve with perturbation
        ptb_sols_pos = _solve_or_cache_3d(ptb_cp_pos, module)
        if module.two_sides:
            # negative perturbed costs
            ptb_cp_neg = cp - module.lambd * dl + module.sigma * noises
            # solve with perturbation
            ptb_sols_neg = _solve_or_cache_3d(ptb_cp_neg, module)
            # get two-side gradient
            grad = (ptb_sols_pos - ptb_sols_neg).mean(dim=1) / (2 * module.lambd + _EPS)
        else:
            # get single side gradient
            grad = (ptb_sols_pos - ptb_sols).mean(dim=1) / (module.lambd + _EPS)
        return grad, None


class adaptiveImplicitMLE(optModule):
    """
    An autograd module for Adaptive Implicit Maximum Likelihood Estimator, which
    adaptively chooses hyperparameter λ and yields an optimal solution in a
    constrained exponential family distribution via Perturb-and-MAP.

    For AI-MLE, it works as black-box combinatorial solvers, in which constraints
    are known and fixed, but the cost vector needs to be predicted from
    contextual data.

    The AI-MLE approximates the gradient of the optimizer smoothly. Thus, it allows us to
    design an algorithm based on stochastic gradient descent.

    Reference: <https://ojs.aaai.org/index.php/AAAI/article/view/26103>
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
            n_samples: number of Monte Carlo samples
            sigma: noise temperature for the input distribution
            distribution: noise distribution
            two_sides: approximate gradient by two-sided perturbation or not
            processes: number of processors, 1 for single-core, 0 for all of cores
            solve_ratio: the ratio of new solutions computed during training
            dataset: the training data
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
        pred_cost, = ctx.saved_tensors
        noises = ctx.noises
        ptb_sols = ctx.ptb_sols
        module = ctx.module
        # convert tensor
        cp = pred_cost.detach()
        dl = grad_output.detach()
        # calculate λ
        dl_norm = torch.norm(dl)
        if dl_norm > 0:
            lambd = module.alpha * torch.norm(cp) / dl_norm
        else:
            lambd = 0.0
        # positive perturbed costs
        ptb_cp_pos = cp + lambd * dl + module.sigma * noises
        # solve with perturbation
        ptb_sols_pos = _solve_or_cache_3d(ptb_cp_pos, module)
        if module.two_sides:
            # negative perturbed costs
            ptb_cp_neg = cp - lambd * dl + module.sigma * noises
            # solve with perturbation
            ptb_sols_neg = _solve_or_cache_3d(ptb_cp_neg, module)
            # get two-side gradient
            grad = (ptb_sols_pos - ptb_sols_neg).mean(dim=1) / (2 * lambd + _EPS)
        else:
            # get single side gradient
            grad = (ptb_sols_pos - ptb_sols).mean(dim=1) / (lambd + _EPS)
        # moving average of the gradient norm
        grad_norm = (grad.abs() > _EPS).float().mean()
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
        # cache branch only fires when solve_ratio < 1, which forces __init__ to populate solpool
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
        ptb_c: perturbed costs, shape (n_samples, batch, vars)
        optmodel: optimization model
        processes: number of processors
        pool: process pool
        solpool: solution pool

    Returns:
        tuple: (solutions shape (batch, n_samples, vars), updated solpool)
    """
    n_samples, ins_num, num_vars = ptb_c.shape
    # flatten (n_samples, batch, vars) → (n_samples * batch, vars)
    flat_c = ptb_c.reshape(-1, num_vars)
    # solve using shared 2D function
    flat_sols, _ = _solve_batch_2d(flat_c, optmodel, processes, pool)
    # reshape (n_samples * batch, vars) → (batch, n_samples, vars)
    ptb_sols = flat_sols.reshape(n_samples, ins_num, num_vars).permute(1, 0, 2)
    # update solution pool on-device, then realign with ptb_c.device
    if solpool is not None:
        sols = ptb_sols.reshape(-1, num_vars)
        solpool = _update_solution_pool(sols, solpool)
        if solpool.device != ptb_c.device:
            solpool = solpool.to(ptb_c.device)
    return ptb_sols, solpool


def _cache_in_pass_3d(
    ptb_c: torch.Tensor,
    optmodel: optModel,
    solpool: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Use solution pool for perturbed 3D costs (n_samples × batch × vars).
    Unlike the 2D version in utils, this handles the extra sample dimension.
    """
    # move solpool to the correct device
    if solpool.device != ptb_c.device:
        solpool = solpool.to(ptb_c.device)
    # compute objective values: (batch, n_samples, pool_size)
    solpool_obj = torch.einsum("nbd,sd->bns", ptb_c, solpool)
    # best solution in pool
    if optmodel.modelSense == EPO.MINIMIZE:
        best_inds = torch.argmin(solpool_obj, dim=2)
    elif optmodel.modelSense == EPO.MAXIMIZE:
        best_inds = torch.argmax(solpool_obj, dim=2)
    else:
        raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
    ptb_sols = solpool[best_inds]
    return ptb_sols, solpool
