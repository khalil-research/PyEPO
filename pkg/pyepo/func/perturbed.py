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
from pyepo.utlis import getArgs
from pyepo.func.utlis import sumGammaDistribution


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
        super().__init__(optmodel, processes, solve_ratio, dataset)
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
        sols = self.ptb.apply(pred_cost, self.optmodel, self.n_samples,
                              self.sigma, self.processes, self.pool, self.rnd,
                              self.solve_ratio, self)
        return sols


class perturbedOptFunc(Function):
    """
    A autograd function for perturbed optimizer
    """

    @staticmethod
    def forward(ctx, pred_cost, optmodel, n_samples, sigma,
                processes, pool, rnd, solve_ratio, module):
        """
        Forward pass for perturbed

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            optmodel (optModel): an PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): the amplitude of the perturbation
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            pool (ProcessPool): process pool object
            rnd (RondomState): numpy random state
            solve_ratio (float): the ratio of new solutions computed during training
            module (optModule): perturbedOpt module

        Returns:
            torch.tensor: solution expectations with perturbation
        """
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach().to("cpu").numpy()
        # sample perturbations
        noises = rnd.normal(0, 1, size=(n_samples, *cp.shape))
        ptb_c = cp + sigma * noises
        # solve with perturbation
        rand_sigma = np.random.uniform()
        if rand_sigma <= solve_ratio:
            ptb_sols = _solve_in_pass(ptb_c, optmodel, processes, pool)
            if solve_ratio < 1:
                sols = ptb_sols.reshape(-1, cp.shape[1])
                # add into solpool
                module.solpool = np.concatenate((module.solpool, sols))
                # remove duplicate
                module.solpool = np.unique(module.solpool, axis=0)
        else:
            ptb_sols = _cache_in_pass(ptb_c, optmodel, module.solpool)
        # solution expectation
        e_sol = ptb_sols.mean(axis=1)
        # convert to tensor
        noises = torch.FloatTensor(noises).to(device)
        ptb_sols = torch.FloatTensor(ptb_sols).to(device)
        e_sol = torch.FloatTensor(e_sol).to(device)
        # save solutions
        ctx.save_for_backward(ptb_sols, noises)
        # add other objects to ctx
        ctx.optmodel = optmodel
        ctx.n_samples = n_samples
        ctx.sigma = sigma
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
        return grad, None, None, None, None, None, None, None, None


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
        super().__init__(optmodel, processes, solve_ratio, dataset)
        # number of samples
        self.n_samples = n_samples
        # perturbation amplitude
        self.sigma = sigma
        # random state
        self.rnd = np.random.RandomState(seed)
        # build optimizer
        self.pfy = perturbedFenchelYoungFunc()

    def forward(self, pred_cost, true_sol, reduction="mean"):
        """
        Forward pass
        """
        loss = self.pfy.apply(pred_cost, true_sol, self.optmodel, self.n_samples,
                              self.sigma, self.processes, self.pool, self.rnd,
                              self.solve_ratio, self)
        # reduction
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(reduction))
        return loss


class perturbedFenchelYoungFunc(Function):
    """
    A autograd function for Fenchel-Young loss using perturbation techniques.
    """

    @staticmethod
    def forward(ctx, pred_cost, true_sol, optmodel, n_samples, sigma,
                processes, pool, rnd, solve_ratio, module):
        """
        Forward pass for perturbed Fenchel-Young loss

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_sol (torch.tensor): a batch of true optimal solutions
            optmodel (optModel): an PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): the amplitude of the perturbation
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            pool (ProcessPool): process pool object
            rnd (RondomState): numpy random state
            solve_ratio (float): the ratio of new solutions computed during training
            module (optModule): perturbedFenchelYoung module

        Returns:
            torch.tensor: solution expectations with perturbation
        """
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach().to("cpu").numpy()
        w = true_sol.detach().to("cpu").numpy()
        # sample perturbations
        noises = rnd.normal(0, 1, size=(n_samples, *cp.shape))
        ptb_c = cp + sigma * noises
        # solve with perturbation
        rand_sigma = np.random.uniform()
        if rand_sigma <= solve_ratio:
            ptb_sols = _solve_in_pass(ptb_c, optmodel, processes, pool)
            if solve_ratio < 1:
                sols = ptb_sols.reshape(-1, cp.shape[1])
                # add into solpool
                module.solpool = np.concatenate((module.solpool, sols))
                # remove duplicate
                module.solpool = np.unique(module.solpool, axis=0)
        else:
            ptb_sols = _cache_in_pass(ptb_c, optmodel, module.solpool)
        # solution expectation
        e_sol = ptb_sols.mean(axis=1)
        # difference
        if optmodel.modelSense == EPO.MINIMIZE:
            diff = w - e_sol
        if optmodel.modelSense == EPO.MAXIMIZE:
            diff = e_sol - w
        # loss
        loss = np.sum(diff**2, axis=1)
        # convert to tensor
        diff = torch.FloatTensor(diff).to(device)
        loss = torch.FloatTensor(loss).to(device)
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
        return grad * grad_output, None, None, None, None, None, None, None, None, None


class implicitMLE(optModule):
    """
    An autograd module for Implicit Maximum Likelihood Estimator, which yield
    an optimal solution in a constrained exponential family distribution via
    Perturb-and-MAP.

    For I-LME, it works as black-box combinatorial solvers, in which constraints
    are known and fixed, but the cost vector need to be predicted from
    contextual data.

    The I-LME approximate gradient of optimizer smoothly. Thus, allows us to
    design an algorithm based on stochastic gradient descent.

    Reference: <https://proceedings.neurips.cc/paper_files/paper/2021/hash/7a430339c10c642c4b2251756fd1b484-Abstract.html>
    """

    def __init__(self, optmodel, n_samples=10, sigma=1.0, lambd=10, processes=1,
                 distribution=sumGammaDistribution(kappa=5), solve_ratio=1,
                 dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): noise temperature for the input distribution
            lambd (float): a hyperparameter for differentiable block-box to control interpolation degree
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            distribution (distribution): noise distribution
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, dataset)
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
        # build I-LME
        self.ilme = implicitMLEFunc()

    def forward(self, pred_cost):
        """
        Forward pass
        """
        sols = self.ilme.apply(pred_cost, self.optmodel, self.n_samples,
                                self.sigma, self.lambd, self.processes,
                                self.pool, self.distribution, self.solve_ratio,
                                self)
        return sols


class implicitMLEFunc(Function):
    """
    A autograd function for Implicit Maximum Likelihood Estimator
    """

    @staticmethod
    def forward(ctx, pred_cost, optmodel, n_samples, sigma, lambd,
                processes, pool, distribution, solve_ratio, module):
        """
        Forward pass for IMLE

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            optmodel (optModel): an PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): noise temperature for the input distribution
            lambd (float): a hyperparameter for differentiable block-box to control interpolation degree
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            pool (ProcessPool): process pool object
            distribution (distribution): noise distribution
            solve_ratio (float): the ratio of new solutions computed during training
            module (optModule): implicitMLE module

        Returns:
            torch.tensor: predicted solutions
        """
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach().to("cpu").numpy()
        # sample perturbations
        noises = distribution.sample(size=(n_samples, *cp.shape))
        ptb_c = cp + sigma * noises
        # solve with perturbation
        rand_sigma = np.random.uniform()
        if rand_sigma <= solve_ratio:
            ptb_sols = _solve_in_pass(ptb_c, optmodel, processes, pool)
            if solve_ratio < 1:
                sols = ptb_sols.reshape(-1, cp.shape[1])
                # add into solpool
                module.solpool = np.concatenate((module.solpool, sols))
                # remove duplicate
                module.solpool = np.unique(module.solpool, axis=0)
        else:
            ptb_sols = _cache_in_pass(ptb_c, optmodel, module.solpool)
        # solution average
        e_sol = ptb_sols.mean(axis=1)
        # convert to tensor
        e_sol = torch.FloatTensor(e_sol).to(device)
        # save
        ctx.save_for_backward(pred_cost)
        # add other objects to ctx
        ctx.noises = noises
        ctx.ptb_sols = ptb_sols
        ctx.lambd = lambd
        ctx.optmodel = optmodel
        ctx.processes = processes
        ctx.pool = pool
        ctx.solve_ratio = solve_ratio
        if solve_ratio < 1:
            ctx.module = module
        ctx.rand_sigma = rand_sigma
        return e_sol

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for IMLE
        """
        pred_cost, = ctx.saved_tensors
        noises = ctx.noises
        ptb_sols = ctx.ptb_sols
        lambd = ctx.lambd
        optmodel = ctx.optmodel
        processes = ctx.processes
        pool = ctx.pool
        solve_ratio = ctx.solve_ratio
        rand_sigma = ctx.rand_sigma
        if solve_ratio < 1:
            module = ctx.module
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach().to("cpu").numpy()
        dl = grad_output.detach().to("cpu").numpy()
        # perturbed costs
        ptb_cq = cp + lambd * dl + noises
        # solve with perturbation
        rand_sigma = np.random.uniform()
        if rand_sigma <= solve_ratio:
            ptb_solsq = _solve_in_pass(ptb_cq, optmodel, processes, pool)
            if solve_ratio < 1:
                sols = ptb_solsq.reshape(-1, cp.shape[1])
                # add into solpool
                module.solpool = np.concatenate((module.solpool, sols))
                # remove duplicate
                module.solpool = np.unique(module.solpool, axis=0)
        else:
            ptb_solsq = _cache_in_pass(ptb_cq, optmodel, module.solpool)
        # get gradient
        grad =  (np.array(ptb_solsq) - ptb_sols).mean(axis=1) / lambd
        # convert to tensor
        grad = torch.FloatTensor(grad).to(device)
        return grad, None, None, None, None, None, None, None, None, None


def _solve_in_pass(ptb_c, optmodel, processes, pool):
    """
    A function to solve optimization in the forward pass
    """
    # number of instance
    n_samples, ins_num = ptb_c.shape[0], ptb_c.shape[1]
    # single-core
    if processes == 1:
        ptb_sols = []
        for i in range(ins_num):
            sols = []
            # per sample
            for j in range(n_samples):
                # solve
                optmodel.setObj(ptb_c[j,i])
                sol, _ = optmodel.solve()
                sols.append(sol)
            ptb_sols.append(sols)
    # multi-core
    else:
        # get class
        model_type = type(optmodel)
        # get args
        args = getArgs(optmodel)
        # parallel computing
        ptb_sols = pool.amap(_solveWithObj4Par, ptb_c.transpose(1,0,2),
                             [args] * ins_num, [model_type] * ins_num).get()
    return np.array(ptb_sols)


def _cache_in_pass(ptb_c, optmodel, solpool):
    """
    A function to use solution pool in the forward/backward pass
    """
    # number of samples & instance
    n_samples, ins_num, _ = ptb_c.shape
    # init sols
    ptb_sols = []
    for j in range(n_samples):
        # best solution in pool
        solpool_obj = ptb_c[j] @ solpool.T
        if optmodel.modelSense == EPO.MINIMIZE:
            ind = np.argmin(solpool_obj, axis=1)
        if optmodel.modelSense == EPO.MAXIMIZE:
            ind = np.argmax(solpool_obj, axis=1)
        ptb_sols.append(solpool[ind])
    return np.array(ptb_sols).transpose(1,0,2)


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
    return sols
