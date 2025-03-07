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
from pyepo.model.mpax import optMpaxModel
from pyepo.utlis import getArgs
from pyepo.func.utlis import sumGammaDistribution

try:
    import jax
    from jax import numpy as jnp
    from mpax import create_lp, r2HPDHG
except ImportError:
    pass


class perturbedOpt(optModule):
    """
    An autograd module for Fenchel-Young loss using perturbation techniques. The
    use of the loss improves the algorithmic by the specific expression of the
    gradients of the loss.

    For the perturbed optimizer, the cost vector needs to be predicted from
    contextual data and are perturbed with Gaussian noise.

    Thus, it allows us to design an algorithm based on stochastic gradient
    descent.

    Reference: <https://papers.nips.cc/paper/2020/hash/6bb56208f672af0dd65451f869fedfd9-Abstract.html>
    """

    def __init__(self, optmodel, n_samples=10, sigma=1.0, processes=1,
                 seed=135, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
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
        # build optimizer
        self.ptb = perturbedOptFunc()

    def forward(self, pred_cost):
        """
        Forward pass
        """
        sols = self.ptb.apply(pred_cost, self)
        return sols


class perturbedOptFunc(Function):
    """
    A autograd function for perturbed optimizer
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
        # convert tenstor
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
    use of the loss improves the algorithmic by the specific expression of the
    gradients of the loss.

    For the perturbed optimizer, the cost vector need to be predicted from
    contextual data and are perturbed with Gaussian noise.

    The Fenchel-Young loss allows to directly optimize a loss between the features
    and solutions with less computation. Thus, allows us to design an algorithm
    based on stochastic gradient descent.

    Reference: <https://papers.nips.cc/paper/2020/hash/6bb56208f672af0dd65451f869fedfd9-Abstract.html>
    """

    def __init__(self, optmodel, n_samples=10, sigma=1.0, processes=1,
                 seed=135, solve_ratio=1, reduction="mean", dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
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
        # build optimizer
        self.pfy = perturbedFenchelYoungFunc()

    def forward(self, pred_cost, true_sol):
        """
        Forward pass
        """
        loss = self.pfy.apply(pred_cost, true_sol, self)
        # reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(self.reduction))
        return loss


class perturbedFenchelYoungFunc(Function):
    """
    A autograd function for Fenchel-Young loss using perturbation techniques.
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
        # convert tenstor
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
    An autograd module for Implicit Maximum Likelihood Estimator, which yield
    an optimal solution in a constrained exponential family distribution via
    Perturb-and-MAP.

    For I-MLE, it works as black-box combinatorial solvers, in which constraints
    are known and fixed, but the cost vector need to be predicted from
    contextual data.

    The I-MLE approximate gradient of optimizer smoothly. Thus, allows us to
    design an algorithm based on stochastic gradient descent.

    Reference: <https://proceedings.neurips.cc/paper_files/paper/2021/hash/7a430339c10c642c4b2251756fd1b484-Abstract.html>
    """

    def __init__(self, optmodel, n_samples=10, sigma=1.0, lambd=10,
                 distribution=sumGammaDistribution(kappa=5), two_sides=False,
                 processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): noise temperature for the input distribution
            lambd (float): a hyperparameter for differentiable block-box to control interpolation degree
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
        self.distribution = distribution
        # symmetric perturbation
        self.two_sides = two_sides
        # build I-LME
        self.imle = implicitMLEFunc()

    def forward(self, pred_cost):
        """
        Forward pass
        """
        sols = self.imle.apply(pred_cost, self)
        return sols


class implicitMLEFunc(Function):
    """
    A autograd function for Implicit Maximum Likelihood Estimator
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
        # convert tenstor
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
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach()
        dl = grad_output.detach()
        # positive perturbed costs
        ptb_cp_pos = cp + module.lambd * dl + noises
        # solve with perturbation
        ptb_sols_pos = _solve_or_cache(ptb_cp_pos, module)
        if module.two_sides:
            # negative perturbed costs
            ptb_cp_neg = cp - module.lambd * dl + noises
            # solve with perturbation
            ptb_sols_neg = _solve_or_cache(ptb_cp_neg, module)
            # get two-side gradient
            grad = (ptb_sols_pos - ptb_sols_neg).mean(dim=1) / (2 * module.lambd)
        else:
            # get single side gradient
            grad = (ptb_sols_pos - ptb_sols).mean(dim=1) / module.lambd
        return grad, None, None


class adaptiveImplicitMLE(optModule):
    """
    An autograd module for Adaptive Implicit Maximum Likelihood Estimator, which
    adaptively choose hyperparameter λ and yield an optimal solution in a
    constrained exponential family distribution via Perturb-and-MAP.

    For AI-MLE, it works as black-box combinatorial solvers, in which constraints
    are known and fixed, but the cost vector need to be predicted from
    contextual data.

    The AI-MLE approximate gradient of optimizer smoothly. Thus, allows us to
    design an algorithm based on stochastic gradient descent.

    Reference: <https://ojs.aaai.org/index.php/AAAI/article/view/26103>
    """

    def __init__(self, optmodel, n_samples=10, sigma=1.0,
                 distribution=sumGammaDistribution(kappa=5), two_sides=False,
                 processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
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
        self.distribution = distribution
        # symmetric perturbation
        self.two_sides = two_sides
        # init adaptive params
        self.alpha = 0 # adaptive magnitude α
        self.grad_norm_avg = 1 # gradient norm estimate
        self.step = 1e-3 # update step for α
        # build I-LME
        self.aimle = adaptiveImplicitMLEFunc()

    def forward(self, pred_cost):
        """
        Forward pass
        """
        sols = self.aimle.apply(pred_cost, self)
        return sols


class adaptiveImplicitMLEFunc(implicitMLEFunc):
    """
    A autograd function for Adaptive Implicit Maximum Likelihood Estimator
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
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach()
        dl = grad_output.detach()
        # calculate λ
        lambd = module.alpha * torch.norm(cp) / torch.norm(dl)
        # positive perturbed costs
        ptb_cp_pos = cp + lambd * dl + noises
        # solve with perturbation
        ptb_sols_pos = _solve_or_cache(ptb_cp_pos, module)
        if module.two_sides:
            # negative perturbed costs
            ptb_cp_neg = cp - lambd * dl + noises
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
        # update α to make grad_norm closer to 1
        if module.grad_norm_avg < 1:
            module.alpha += module.step
        else:
            module.alpha -= module.step
        return grad, None, None


def _solve_or_cache(ptb_c, module):
    # solve optimization
    if np.random.uniform() <= module.solve_ratio:
        ptb_sols = _solve_in_pass(ptb_c, module.optmodel, module.processes, module.pool)
        if module.solve_ratio < 1:
            sols = ptb_sols.view(-1, ptb_sols.shape[2])
            # add into solpool
            module._update_solution_pool(sols)
    # best cached solution
    else:
        ptb_sols = _cache_in_pass(ptb_c, module.optmodel, module.solpool)
    return ptb_sols


def _solve_in_pass(ptb_c, optmodel, processes, pool):
    """
    A function to solve optimization in the forward pass
    """
    # get device
    device = ptb_c.device
    # number and size of instance
    n_samples, ins_num, num_vars = ptb_c.shape
    # MPAX batch solving
    if isinstance(optmodel, optMpaxModel):
        # flat
        ptb_c = ptb_c.reshape(-1, num_vars)
        # get params
        optmodel.setObj(ptb_c)
        ptb_c = optmodel.c
        # batch solving
        ptb_sols, _ = optmodel.batch_optimize(ptb_c)
        # convert to torch
        ptb_sols = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(ptb_sols)).to(device)
        # reshape
        ptb_sols = ptb_sols.view(n_samples, ins_num, num_vars).permute(1, 0, 2)
    # single-core
    elif processes == 1:
        ptb_sols = torch.zeros((ins_num, n_samples, num_vars), dtype=torch.float32, device=device)
        for i in range(ins_num):
            # per sample
            for j in range(n_samples):
                # solve
                optmodel.setObj(ptb_c[j,i])
                sol, _ = optmodel.solve()
                ptb_sols[i,j] = torch.as_tensor(sol, dtype=torch.float32, device=device)
    # multi-core
    else:
        # get class
        model_type = type(optmodel)
        # get args
        args = getArgs(optmodel)
        # parallel computing
        res = pool.amap(_solveWithObj4Par, ptb_c.permute(1, 0, 2),
                        [args] * ins_num, [model_type] * ins_num).get()
        # get solution
        ptb_sols = torch.stack(res, dim=0).to(device)
    return ptb_sols


def _cache_in_pass(ptb_c, optmodel, solpool):
    """
    A function to use solution pool in the forward/backward pass
    """
    # get device
    device = ptb_c.device
    # solpool is on the same device
    if solpool.device != device:
        solpool = solpool.to(device)
    # compute objective values for all perturbations
    solpool_obj = torch.einsum("nbd,sd->bns", ptb_c, solpool)
    # best solution in pool
    if optmodel.modelSense == EPO.MINIMIZE:
        best_inds = torch.argmin(solpool_obj, dim=2)
    elif optmodel.modelSense == EPO.MAXIMIZE:
        best_inds = torch.argmax(solpool_obj, dim=2)
    else:
        raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
    ptb_sols = solpool[best_inds]
    return ptb_sols


def _solveWithObj4Par(perturbed_costs, args, model_type):
    """
    A global function to solve function in parallel processors

    Args:
        perturbed_costs (np.ndarray): costsof objective function with perturbation
        args (dict): optModel args
        model_type (ABCMeta): optModel class type

    Returns:
        list: optimal solution
    """
    # rebuild model
    optmodel = model_type(**args)
    # per sample
    sols = []
    for cost in perturbed_costs:
        # set obj
        optmodel.setObj(cost)
        # solve
        sol, _ = optmodel.solve()
        sols.append(sol)
    # to tensor
    sols = torch.tensor(sols, dtype=torch.float32)
    return sols
