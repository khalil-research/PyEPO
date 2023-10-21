#!/usr/bin/env python
# coding: utf-8
"""
Differentiable Black-box optimization function
"""

import numpy as np
import torch
from torch.autograd import Function

from pyepo.func.abcmodule import optModule
from pyepo import EPO
from pyepo.func.utlis import _solveWithObj4Par, _solve_in_pass, _cache_in_pass


class blackboxOpt(optModule):
    """
    An autograd module for differentiable black-box optimizer, which yield
    an optimal solution and derive a gradient.

    For differentiable block-box, the objective function is linear and
    constraints are known and fixed, but the cost vector needs to be predicted
    from contextual data.

    The block-box approximates the gradient of the optimizer by interpolating
    the loss function. Thus, it allows us to design an algorithm based on
    stochastic gradient descent.

    Reference: <https://arxiv.org/abs/1912.02175>
    """

    def __init__(self, optmodel, lambd=10, processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            lambd (float): a hyperparameter for differentiable block-box to control interpolation degree
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, dataset)
        # smoothing parameter
        if lambd <= 0:
            raise ValueError("lambda is not positive.")
        self.lambd = lambd
        # build blackbox optimizer
        self.dbb = blackboxOptFunc()

    def forward(self, pred_cost):
        """
        Forward pass
        """
        sols = self.dbb.apply(pred_cost, self.lambd, self.optmodel,
                              self.processes, self.pool, self.solve_ratio, self)
        return sols


class blackboxOptFunc(Function):
    """
    A autograd function for differentiable black-box optimizer
    """

    @staticmethod
    def forward(ctx, pred_cost, lambd, optmodel, processes, pool, solve_ratio, module):
        """
        Forward pass for DBB

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            lambd (float): a hyperparameter for differentiable block-box to control interpolation degree
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            pool (ProcessPool): process pool object
            solve_ratio (float): the ratio of new solutions computed during training
            module (optModule): blackboxOpt module

        Returns:
            torch.tensor: predicted solutions
        """
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach().to("cpu").numpy()
        # solve
        rand_sigma = np.random.uniform()
        if rand_sigma <= solve_ratio:
            sol, _ = _solve_in_pass(cp, optmodel, processes, pool)
            if solve_ratio < 1:
                # add into solpool
                module.solpool = np.concatenate((module.solpool, sol))
                # remove duplicate
                module.solpool = np.unique(module.solpool, axis=0)
        else:
            sol, _ = _cache_in_pass(cp, optmodel, module.solpool)
        # convert to tensor
        sol = np.array(sol)
        pred_sol = torch.FloatTensor(sol).to(device)
        # save
        ctx.save_for_backward(pred_cost, pred_sol)
        # add other objects to ctx
        ctx.lambd = lambd
        ctx.optmodel = optmodel
        ctx.processes = processes
        ctx.pool = pool
        ctx.solve_ratio = solve_ratio
        if solve_ratio < 1:
            ctx.module = module
        ctx.rand_sigma = rand_sigma
        return pred_sol

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for DBB
        """
        pred_cost, pred_sol = ctx.saved_tensors
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
        wp = pred_sol.detach().to("cpu").numpy()
        dl = grad_output.detach().to("cpu").numpy()
        # perturbed costs
        cq = cp + lambd * dl
        # solve
        if rand_sigma <= solve_ratio:
            sol, _ = _solve_in_pass(cq, optmodel, processes, pool)
            if solve_ratio < 1:
                # add into solpool
                module.solpool = np.concatenate((module.solpool, sol))
                # remove duplicate
                module.solpool = np.unique(module.solpool, axis=0)
        else:
            sol, _ = _cache_in_pass(cq, optmodel, module.solpool)
        # get gradient
        grad = (np.array(sol) - wp) / lambd
        # convert to tensor
        grad = torch.FloatTensor(grad).to(device)
        return grad, None, None, None, None, None, None


class negativeIdentity(optModule):
    """
    An autograd module for the differentiable optimizer, which yields optimal a
    solution and use negative identity as a gradient on the backward pass.

    For negative identity backpropagation, the objective function is linear and
    constraints are known and fixed, but the cost vector needs to be predicted
    from contextual data.

    If the interpolation hyperparameter λ aligns with an appropriate step size,
    then the identity update is equivalent to DBB. However, the identity update
    does not require an additional call to the solver during the backward pass
    and tuning an additional hyperparameter λ.

    Reference: <https://arxiv.org/abs/2205.15213>
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, dataset)
        # build blackbox optimizer
        self.nid = negativeIdentityFunc()

    def forward(self, pred_cost):
        """
        Forward pass
        """
        sols = self.nid.apply(pred_cost, self.optmodel, self.processes,
                              self.pool, self.solve_ratio, self)
        return sols


class negativeIdentityFunc(Function):
    """
    A autograd function for differentiable black-box optimizer
    """

    @staticmethod
    def forward(ctx, pred_cost, optmodel, processes, pool, solve_ratio, module):
        """
        Forward pass for NID

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            pool (ProcessPool): process pool object
            solve_ratio (float): the ratio of new solutions computed during training
            module (optModule): blackboxOpt module

        Returns:
            torch.tensor: predicted solutions
        """
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach().to("cpu").numpy()
        # solve
        rand_sigma = np.random.uniform()
        if rand_sigma <= solve_ratio:
            sol, _ = _solve_in_pass(cp, optmodel, processes, pool)
            if solve_ratio < 1:
                # add into solpool
                module.solpool = np.concatenate((module.solpool, sol))
                # remove duplicate
                module.solpool = np.unique(module.solpool, axis=0)
        else:
            sol, _ = _cache_in_pass(cp, optmodel, module.solpool)
        # convert to tensor
        pred_sol = torch.FloatTensor(np.array(sol)).to(device)
        # add other objects to ctx
        ctx.optmodel = optmodel
        return pred_sol

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for NID
        """
        optmodel = ctx.optmodel
        # get device
        device = grad_output.device
        # identity matrix
        I = torch.eye(grad_output.shape[1]).to(device)
        if optmodel.modelSense == EPO.MINIMIZE:
            grad = - I
        if optmodel.modelSense == EPO.MAXIMIZE:
            grad = I
        return grad_output @ grad, None, None, None, None, None
