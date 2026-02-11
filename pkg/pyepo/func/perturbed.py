#!/usr/bin/env python
# coding: utf-8
"""
Perturbed optimization function
"""

import numpy as np
import torch
from torch.autograd import Function

from pyepo import EPO
from pyepo.func.abcmodule import optModule
from pyepo.func.utils import (
    _solve_batch as _solve_batch_2d,
    _update_solution_pool,
    sumGammaDistribution,
)


class perturbedOpt(optModule):
    """
    An autograd module for Fenchel-Young loss using perturbation techniques. The
    use of the loss improves the algorithm by the specific expression of the
    gradients of the loss.

    For the perturbed optimizer, the cost vector needs to be predicted from
    contextual data and is perturbed with Gaussian noise.

    Thus, it allows us to design an algorithm based on stochastic gradient
    descent.

    Reference: <https://papers.nips.cc/paper/2020/hash/6bb56208f672af0dd65451f869fedfd9-Abstract.html>
    """

    def __init__(self, optmodel, n_samples=10, sigma=1.0, processes=1,
                 seed=135, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): a PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): the amplitude of the perturbation
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            seed (int): random state seed
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, dataset=dataset)
        # number of samples
        self.n_samples = n_samples
        # perturbation amplitude
        self.sigma = sigma
        # random state
        self.rnd = np.random.RandomState(seed)

    def forward(self, pred_cost):
        """
        Forward pass
        """
        sols = perturbedOptFunc.apply(pred_cost, self)
        return sols


class perturbedOptFunc(Function):
    """
    An autograd function for perturbed optimizer
    """

    @staticmethod
    def forward(ctx, pred_cost, module):
        """
        Forward pass for perturbed

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            module (optModule): perturbedOpt module

        Returns:
            torch.tensor: solution expectations with perturbation
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach()
        # sample perturbations
        noises = module.rnd.normal(0, 1, size=(module.n_samples, *cp.shape))
        noises = torch.from_numpy(noises).to(device, dtype=torch.float32)
        ptb_c = cp + module.sigma * noises
        # solve with perturbation
        ptb_sols = _solve_or_cache(ptb_c, module)
        # solution expectation
        e_sol = ptb_sols.mean(dim=1)
        # save solutions
        ctx.save_for_backward(ptb_sols, noises)
        # add other objects to ctx
        ctx.optmodel = module.optmodel
        ctx.n_samples = module.n_samples
        ctx.sigma = module.sigma
        return e_sol

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for perturbed
        """
        ptb_sols, noises = ctx.saved_tensors
        optmodel = ctx.optmodel
        n_samples = ctx.n_samples
        sigma = ctx.sigma
        grad = torch.einsum("nbd,bn->bd",
                            noises,
                            torch.einsum("bnd,bd->bn", ptb_sols, grad_output))
        grad /= n_samples * sigma
        return grad, None


class perturbedFenchelYoung(optModule):
    """
    An autograd module for Fenchel-Young loss using perturbation techniques. The
    use of the loss improves the algorithm by the specific expression of the
    gradients of the loss.

    For the perturbed optimizer, the cost vector needs to be predicted from
    contextual data and is perturbed with Gaussian noise.

    The Fenchel-Young loss allows directly optimizing a loss between the features
    and solutions with less computation. Thus, it allows us to design an algorithm
    based on stochastic gradient descent.

    Reference: <https://papers.nips.cc/paper/2020/hash/6bb56208f672af0dd65451f869fedfd9-Abstract.html>
    """

    def __init__(self, optmodel, n_samples=10, sigma=1.0, processes=1,
                 seed=135, solve_ratio=1, reduction="mean", dataset=None):
        """
        Args:
            optmodel (optModel): a PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): the amplitude of the perturbation
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            seed (int): random state seed
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (None/optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        # number of samples
        self.n_samples = n_samples
        # perturbation amplitude
        self.sigma = sigma
        # random state
        self.rnd = np.random.RandomState(seed)

    def forward(self, pred_cost, true_sol):
        """
        Forward pass
        """
        loss = perturbedFenchelYoungFunc.apply(pred_cost, true_sol, self)
        return self._reduce(loss)


class perturbedFenchelYoungFunc(Function):
    """
    An autograd function for Fenchel-Young loss using perturbation techniques.
    """

    @staticmethod
    def forward(ctx, pred_cost, true_sol, module):
        """
        Forward pass for perturbed Fenchel-Young loss

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_sol (torch.tensor): a batch of true optimal solutions
            module (optModule): perturbedFenchelYoung module

        Returns:
            torch.tensor: solution expectations with perturbation
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach()
        w = true_sol.detach()
        # sample perturbations
        noises = module.rnd.normal(0, 1, size=(module.n_samples, *cp.shape))
        noises = torch.from_numpy(noises).to(device, dtype=torch.float32)
        ptb_c = cp + module.sigma * noises
        # solve with perturbation
        ptb_sols = _solve_or_cache(ptb_c, module)
        # solution expectation
        e_sol = ptb_sols.mean(dim=1)
        # difference
        if module.optmodel.modelSense == EPO.MINIMIZE:
            diff = w - e_sol
        elif module.optmodel.modelSense == EPO.MAXIMIZE:
            diff = e_sol - w
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        # loss
        loss = torch.sum(diff**2, dim=1)
        # save solutions
        ctx.save_for_backward(diff)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for perturbed Fenchel-Young loss
        """
        grad, = ctx.saved_tensors
        grad_output = torch.unsqueeze(grad_output, dim=-1)
        return grad * grad_output, None, None


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

    def __init__(self, optmodel, n_samples=10, sigma=1.0, lambd=10,
                 distribution=None, two_sides=False,
                 processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): a PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): noise temperature for the input distribution
            lambd (float): a hyperparameter for differentiable black-box to control interpolation degree
            distribution (distribution): noise distribution
            two_sides (bool): approximate gradient by two-sided perturbation or not
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
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

    def forward(self, pred_cost):
        """
        Forward pass
        """
        sols = implicitMLEFunc.apply(pred_cost, self)
        return sols


class implicitMLEFunc(Function):
    """
    An autograd function for Implicit Maximum Likelihood Estimator
    """

    @staticmethod
    def forward(ctx, pred_cost, module):
        """
        Forward pass for IMLE

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            module (optModule): implicitMLE module

        Returns:
            torch.tensor: predicted solutions
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach()
        # sample perturbations
        noises = module.distribution.sample(size=(module.n_samples, *cp.shape))
        noises = torch.from_numpy(noises).to(device, dtype=torch.float32)
        ptb_c = cp + module.sigma * noises
        # solve with perturbation
        ptb_sols = _solve_or_cache(ptb_c, module)
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
    def backward(ctx, grad_output):
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
        ptb_sols_pos = _solve_or_cache(ptb_cp_pos, module)
        if module.two_sides:
            # negative perturbed costs
            ptb_cp_neg = cp - module.lambd * dl + module.sigma * noises
            # solve with perturbation
            ptb_sols_neg = _solve_or_cache(ptb_cp_neg, module)
            # get two-side gradient
            grad = (ptb_sols_pos - ptb_sols_neg).mean(dim=1) / (2 * module.lambd)
        else:
            # get single side gradient
            grad = (ptb_sols_pos - ptb_sols).mean(dim=1) / module.lambd
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

    def __init__(self, optmodel, n_samples=10, sigma=1.0,
                 distribution=None, two_sides=False,
                 processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): a PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): noise temperature for the input distribution
            distribution (distribution): noise distribution
            two_sides (bool): approximate gradient by two-sided perturbation or not
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
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
        self.alpha = 1.0 # adaptive magnitude α
        self.grad_norm_avg = 1 # gradient sparsity estimate
        self.step = 1e-3 # update step for α

    def forward(self, pred_cost):
        """
        Forward pass
        """
        sols = adaptiveImplicitMLEFunc.apply(pred_cost, self)
        return sols


class adaptiveImplicitMLEFunc(implicitMLEFunc):
    """
    An autograd function for Adaptive Implicit Maximum Likelihood Estimator
    """
    @staticmethod
    def backward(ctx, grad_output):
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
        ptb_sols_pos = _solve_or_cache(ptb_cp_pos, module)
        if module.two_sides:
            # negative perturbed costs
            ptb_cp_neg = cp - lambd * dl + module.sigma * noises
            # solve with perturbation
            ptb_sols_neg = _solve_or_cache(ptb_cp_neg, module)
            # get two-side gradient
            grad = (ptb_sols_pos - ptb_sols_neg).mean(dim=1) / (2 * lambd + 1e-7)
        else:
            # get single side gradient
            grad = (ptb_sols_pos - ptb_sols).mean(dim=1) / (lambd + 1e-7)
        # moving average of the gradient norm
        grad_norm = (grad.abs() > 1e-7).float().mean()
        module.grad_norm_avg = 0.9 * module.grad_norm_avg + 0.1 * grad_norm
        # update α to target gradient
        if module.grad_norm_avg < 1:
            module.alpha += module.step
        else:
            module.alpha = max(0.0, module.alpha - module.step)
        return grad, None


def _solve_or_cache(ptb_c, module):
    """
    Solve or use cached solutions for perturbed costs (3D: n_samples × batch × vars).
    Delegates to the shared 2D functions in utils after flattening.
    """
    optmodel = module.optmodel
    processes = module.processes
    pool = module.pool
    solpool = module.solpool
    solset = module._solset
    if np.random.uniform() <= module.solve_ratio:
        ptb_sols, solpool = _solve_in_pass(ptb_c, optmodel, processes, pool, solpool, solset)
    else:
        ptb_sols, solpool = _cache_in_pass(ptb_c, optmodel, solpool)
    module.solpool = solpool
    return ptb_sols


def _solve_in_pass(ptb_c, optmodel, processes, pool, solpool=None, solset=None):
    """
    Solve optimization for perturbed 3D costs and update solution pool.

    Args:
        ptb_c (torch.tensor): perturbed costs, shape (n_samples, batch, vars)
        optmodel (optModel): optimization model
        processes (int): number of processors
        pool: process pool
        solpool (torch.tensor): solution pool
        solset (set): hash set for deduplication

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
    # update solution pool and ensure correct device
    if solpool is not None:
        sols = ptb_sols.reshape(-1, num_vars)
        solpool = _update_solution_pool(sols, solpool, solset)
        if solpool.device != ptb_c.device:
            solpool = solpool.to(ptb_c.device)
    return ptb_sols, solpool


def _cache_in_pass(ptb_c, optmodel, solpool):
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
