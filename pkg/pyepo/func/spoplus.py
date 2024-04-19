#!/usr/bin/env python
# coding: utf-8
"""
SPO+ Loss function
"""

import numpy as np
import torch
from torch.autograd import Function

from pyepo import EPO
from pyepo.func.abcmodule import optModule
from pyepo.func.utlis import _solve_or_cache

class SPOPlus(optModule):
    """
    An autograd module for SPO+ Loss, as a surrogate loss function of SPO Loss,
    which measures the decision error of the optimization problem.

    For SPO/SPO+ Loss, the objective function is linear and constraints are
    known and fixed, but the cost vector needs to be predicted from contextual
    data.

    The SPO+ Loss is convex with subgradient. Thus, it allows us to design an
    algorithm based on stochastic gradient descent.

    Reference: <https://doi.org/10.1287/mnsc.2020.3922>
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, reduction="mean", dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (None/optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        # build carterion
        self.spop = SPOPlusFunc()

    def forward(self, pred_cost, true_cost, true_sol, true_obj):
        """
        Forward pass
        """
        loss = self.spop.apply(pred_cost, true_cost, true_sol, true_obj, self)
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


class SPOPlusFunc(Function):
    """
    A autograd function for SPO+ Loss
    """

    @staticmethod
    def forward(ctx, pred_cost, true_cost, true_sol, true_obj, module):
        """
        Forward pass for SPO+

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_cost (torch.tensor): a batch of true values of the cost
            true_sol (torch.tensor): a batch of true optimal solutions
            true_obj (torch.tensor): a batch of true optimal objective values
            module (optModule): SPOPlus modeul

        Returns:
            torch.tensor: SPO+ loss
        """
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach().to("cpu").numpy()
        c = true_cost.detach().to("cpu").numpy()
        w = true_sol.detach().to("cpu").numpy()
        z = true_obj.detach().to("cpu").numpy()
        # check sol
        #_check_sol(c, w, z)
        # solve
        sol, obj = _solve_or_cache(2 * cp - c, module)
        # calculate loss
        loss = []
        for i in range(len(cp)):
            loss.append(- obj[i] + 2 * np.dot(cp[i], w[i]) - z[i])
        # sense
        if module.optmodel.modelSense == EPO.MINIMIZE:
            loss = np.array(loss)
        if module.optmodel.modelSense == EPO.MAXIMIZE:
            loss = - np.array(loss)
        # convert to tensor
        loss = torch.FloatTensor(loss).to(device)
        sol = torch.FloatTensor(sol).to(device)
        # save solutions
        ctx.save_for_backward(true_sol, sol)
        # add other objects to ctx
        ctx.optmodel = module.optmodel
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for SPO+
        """
        w, wq = ctx.saved_tensors
        optmodel = ctx.optmodel
        if optmodel.modelSense == EPO.MINIMIZE:
            grad = 2 * (w - wq)
        if optmodel.modelSense == EPO.MAXIMIZE:
            grad = 2 * (wq - w)
        return grad_output * grad, None, None, None, None
