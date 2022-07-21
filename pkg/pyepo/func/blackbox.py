#!/usr/bin/env python
# coding: utf-8
"""
Differentiable Black-box optimization function
"""

import multiprocessing as mp

import numpy as np
import torch
from pathos.multiprocessing import ProcessingPool
from torch.autograd import Function
from torch import nn

from pyepo import EPO
from pyepo.data.dataset import optDataset
from pyepo.model.opt import optModel
from pyepo.utlis import getArgs


class blackboxOpt(nn.Module):
    """
    A autograd module for differentiable black-box optimizer, which yield
    optimal a solution and derive a gradient.

    For differentiable block-box, the objective function is linear and
    constraints are known and fixed, but the cost vector need to be predicted
    from contextual data.

    The block-box approximate gradient of optimizer smoothly. Thus, allows us to
    design an algorithm based on stochastic gradient descent.
    """

    def __init__(self, optmodel, lambd=10, processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            lambd (float): a hyperparameter for differentiable block-box to contral interpolation degree
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__()
        # optimization model
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel.")
        self.optmodel = optmodel
        # smoothing parameter
        if lambd <= 0:
            raise ValueError("lambda is not positive.")
        self.lambd = lambd
        # num of processors
        if processes not in range(mp.cpu_count()+1):
            raise ValueError("Invalid processors number {}, only {} cores.".
                format(processes, mp.cpu_count()))
        self.processes = processes
        print("Num of cores: {}".format(self.processes))
        # solution pool
        self.solve_ratio = solve_ratio
        if (self.solve_ratio < 0) or (self.solve_ratio > 1):
            raise ValueError("Invalid solving ratio {}. It should be between 0 and 1.".
                format(self.solve_ratio))
        self.solpool = None
        if self.solve_ratio < 1: # init solution pool
            if not isinstance(dataset, optDataset): # type checking
                raise TypeError("dataset is not an optDataset")
            self.solpool = dataset.sols.copy()
        # build carterion
        self.dbb = blackboxOptFunc()

    def forward(self, pred_cost):
        """
        Forward pass
        """
        loss = self.dbb.apply(pred_cost, self.lambd, self.optmodel,
                              self.processes, self.solve_ratio, self)
        return loss


class blackboxOptFunc(Function):
    """
    A autograd function for differentiable black-box optimizer
    """

    @staticmethod
    def forward(ctx, pred_cost, lambd, optmodel, processes, solve_ratio, module):
        """
        Forward pass for DBB

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            lambd (float): a hyperparameter for differentiable block-box to contral interpolation degree
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            module (nn.Module): blackboxOpt modeul

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
            sol = _solve_in_pass(cp, optmodel, processes)
            if solve_ratio < 1:
                module.solpool = np.concatenate((module.solpool, sol))
        else:
            sol = _cache_in_pass(cp, optmodel, module.solpool)
        # convert to tensor
        sol = np.array(sol)
        pred_sol = torch.FloatTensor(sol).to(device)
        # save
        ctx.save_for_backward(pred_cost, pred_sol)
        # add other objects to ctx
        ctx.lambd = lambd
        ctx.optmodel = optmodel
        ctx.processes = processes
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
        solve_ratio = ctx.solve_ratio
        rand_sigma = ctx.rand_sigma
        if solve_ratio < 1:
            module = ctx.module
        ins_num = len(pred_cost)
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
            sol = _solve_in_pass(cq, optmodel, processes)
            if solve_ratio < 1:
                module.solpool = np.concatenate((module.solpool, sol))
        else:
            sol = _cache_in_pass(cq, optmodel, module.solpool)
        # get gradient
        grad = []
        for i in range(len(sol)):
            grad.append((sol[i] - wp[i]) / lambd)
        # convert to tensor
        grad = np.array(grad)
        grad = torch.FloatTensor(grad).to(device)
        return grad, None, None, None, None, None


def _solve_in_pass(cp, optmodel, processes):
    """
    A function to solve optimization in the forward/backward pass
    """
    # number of instance
    ins_num = len(cp)
    # single-core
    if processes == 1:
        sol = []
        for i in range(ins_num):
            # solve
            optmodel.setObj(cp[i])
            solp, _ = optmodel.solve()
            sol.append(solp)
    # multi-core
    else:
        # number of processes
        processes = mp.cpu_count() if not processes else processes
        # get class
        model_type = type(optmodel)
        # get args
        args = getArgs(optmodel)
        # parallel computing
        with ProcessingPool(processes) as pool:
            sol = pool.amap(_solveWithObj4Par, cp, [args] * ins_num,
                            [model_type] * ins_num).get()
    return sol


def _cache_in_pass(c, optmodel, solpool):
    """
    A function to use solution pool in the forward/backward pass
    """
    # number of instance
    ins_num = len(c)
    # best solution in pool
    solpool_obj = c @ solpool.T
    if optmodel.modelSense == EPO.MINIMIZE:
        ind = np.argmin(solpool_obj, axis=1)
    if optmodel.modelSense == EPO.MAXIMIZE:
        ind = np.argmax(solpool_obj, axis=1)
    sol = solpool[ind]
    return sol


def _solveWithObj4Par(cost, args, model_type):
    """
    A global function to solve function in parallel processors

    Args:
        cost (np.ndarray): cost of objective function
        args (dict): optModel args
        model_type (ABCMeta): optModel class type

    Returns:
        list: optimal solution
    """
    # rebuild model
    optmodel = model_type(**args)
    # set obj
    optmodel.setObj(cost)
    # solve
    sol, _ = optmodel.solve()
    return sol
