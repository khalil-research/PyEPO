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
from pyepo.func.utlis import _solve_or_cache


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
        super().__init__(optmodel, processes, solve_ratio, dataset=dataset)
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
        sols = self.dbb.apply(pred_cost, self)
        return sols


class blackboxOptFunc(Function):
    """
    A autograd function for differentiable black-box optimizer
    """

    @staticmethod
    def forward(ctx, pred_cost, module):
        """
        Forward pass for DBB

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            module (optModule): blackboxOpt module

        Returns:
            torch.tensor: predicted solutions
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach()
        # solve
        sol, _ = _solve_or_cache(cp, module)
        # save
        ctx.save_for_backward(pred_cost, sol)
        # add other objects to ctx
        ctx.lambd = module.lambd
        ctx.module = module
        return sol

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for DBB
        """
        pred_cost, pred_sol = ctx.saved_tensors
        lambd = ctx.lambd
        module = ctx.module
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach()
        wp = pred_sol.detach()
        dl = grad_output.detach()
        # perturbed costs
        cq = cp + lambd * dl
        # solve
        sol, _ = _solve_or_cache(cq, module)
        # get gradient
        grad = (sol - wp) / module.lambd
        # convert to tensor
        grad = torch.as_tensor(grad, dtype=torch.float, device=device)
        return grad, None


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
        super().__init__(optmodel, processes, solve_ratio, dataset=dataset)
        # build blackbox optimizer
        self.nid = negativeIdentityFunc()

    def forward(self, pred_cost):
        """
        Forward pass
        """
        sols = self.nid.apply(pred_cost, self)
        return sols


class negativeIdentityFunc(Function):
    """
    A autograd function for differentiable black-box optimizer
    """

    @staticmethod
    def forward(ctx, pred_cost, module):
        """
        Forward pass for NID

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            module (optModule): blackboxOpt module

        Returns:
            torch.tensor: predicted solutions
        """
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach()
        # solve
        sol, _ = _solve_or_cache(cp, module)
        # add other objects to ctx
        ctx.optmodel = module.optmodel
        return sol

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
        elif optmodel.modelSense == EPO.MAXIMIZE:
            grad = I
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        return grad_output @ grad, None
