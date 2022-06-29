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

from pyepo import EPO
from pyepo.data.dataset import optDataset
from pyepo.model.opt import optModel
from pyepo.utlis import getArgs


def solveWithObj4Par(cost, args, model_type):
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


class blackboxOpt(Function):
    """
    A autograd function for differentiable black-box optimizer, which yield
    optimal a solution and derive a gradient.

    For differentiable black-box, the objective function is linear and
    constraints are known and fixed, but the cost vector need to be predicted
    from contextual data.

    The black-box approximate gradient of optimizer smoothly. Thus, allows us to
    design an algorithm based on stochastic gradient descent.
    """

    def __init__(self, optmodel, lambd=10, processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            lambd (float): a hyperparameter for differentiable black-box to contral interpolation degree
            processes (int): number of processors, 1 for single-core, 0 for all of cores
        """
        super().__init__()
        # optimization model
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel.")
        global _PyEPO_FUNC_DBB_OPTMODEL
        _PyEPO_FUNC_DBB_OPTMODEL = optmodel
        # smoothing parameter
        if lambd <= 0:
            raise ValueError("lambda is not positive.")
        global _PyEPO_FUNC_DBB_LAMBDA
        _PyEPO_FUNC_DBB_LAMBDA = lambd
        # num of processors
        if processes not in range(mp.cpu_count()+1):
            raise ValueError("Invalid processors number {}, only {} cores.".
                format(processes, mp.cpu_count()))
        global _PyEPO_FUNC_DBB_PROCESSES
        _PyEPO_FUNC_DBB_PROCESSES = processes
        print("Num of cores: {}".format(_PyEPO_FUNC_DBB_PROCESSES))
        # solution pool
        if (solve_ratio < 0) or (solve_ratio > 1):
            raise ValueError("Invalid solving ratio {}. It should be between 0 and 1.".
                format(solve_ratio))
        global _PyEPO_FUNC_DBB_PSOLVE
        _PyEPO_FUNC_DBB_PSOLVE = solve_ratio
        if solve_ratio < 1: # init solution pool
            if not isinstance(dataset, optDataset): # type checking
                raise TypeError("dataset is not an optDataset")
            global _PyEPO_FUNC_DBB_POOL
            _PyEPO_FUNC_DBB_POOL = dataset.sols.copy()

    @staticmethod
    def forward(ctx, pred_cost):
        """
        Forward pass in neural network.

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost

        Returns:
            torch.tensor: predicted solutions
        """
        # get global
        global _PyEPO_FUNC_DBB_PSOLVE
        solve_ratio = _PyEPO_FUNC_DBB_PSOLVE
        if solve_ratio < 1:
            global _PyEPO_FUNC_DBB_POOL
            pool = _PyEPO_FUNC_DBB_POOL
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.to("cpu").numpy()
        # solve
        rand_sigma = np.random.uniform()
        if rand_sigma <= solve_ratio:
            sol = _solve_in_pass(cp)
            if solve_ratio < 1:
                _PyEPO_FUNC_DBB_POOL = np.concatenate((pool, sol))
        else:
            sol = _cache_in_pass(cp)
        # convert to tensor
        sol = np.array(sol)
        pred_sol = torch.FloatTensor(sol).to(device)
        rand_sigma = np.array(rand_sigma)
        rand_sigma = torch.FloatTensor(rand_sigma).to(device)
        # save
        ctx.save_for_backward(pred_cost, pred_sol, rand_sigma)
        return pred_sol

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass in neural network
        """
        pred_cost, pred_sol, rand_sigma = ctx.saved_tensors
        ins_num = len(pred_cost)
        # get global
        global _PyEPO_FUNC_DBB_LAMBDA
        lambd = _PyEPO_FUNC_DBB_LAMBDA
        global _PyEPO_FUNC_DBB_PSOLVE
        solve_ratio = _PyEPO_FUNC_DBB_PSOLVE
        if solve_ratio < 1:
            global _PyEPO_FUNC_DBB_POOL
            pool = _PyEPO_FUNC_DBB_POOL
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.to("cpu").numpy()
        wp = pred_sol.to("cpu").numpy()
        dl = grad_output.to("cpu").numpy()
        rand_sigma = rand_sigma.data.cpu().numpy()
        # perturbed costs
        cq = cp + lambd * dl
        # solve
        if rand_sigma <= solve_ratio:
            sol = _solve_in_pass(cq)
            if solve_ratio < 1:
                _PyEPO_FUNC_DBB_POOL = np.concatenate((pool, sol))
        else:
            sol = _cache_in_pass(cq)
        # get gradient
        grad = []
        for i in range(len(sol)):
            grad.append((sol[i] - wp[i]) / lambd)
        # convert to tensor
        grad = np.array(grad)
        grad = torch.FloatTensor(grad).to(device)
        return grad


def _solve_in_pass(cp):
    """
    A function to solve optimization in the forward/backward pass
    """
    # number of instance
    ins_num = len(cp)
    # get global
    global _PyEPO_FUNC_DBB_OPTMODEL
    optmodel = _PyEPO_FUNC_DBB_OPTMODEL
    global _PyEPO_FUNC_DBB_PROCESSES
    processes = _PyEPO_FUNC_DBB_PROCESSES
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
            sol = pool.amap(solveWithObj4Par, cp, [args] * ins_num,
                            [model_type] * ins_num).get()
    return sol


def _cache_in_pass(c):
    """
    A function to use solution pool in the forward/backward pass
    """
    # number of instance
    ins_num = len(c)
    # get global
    global _PyEPO_FUNC_DBB_POOL
    pool = _PyEPO_FUNC_DBB_POOL
    global _PyEPO_FUNC_DBB_OPTMODEL
    optmodel = _PyEPO_FUNC_DBB_OPTMODEL
    # best solution in pool
    pool_obj = c @ pool.T
    if optmodel.modelSense == EPO.MINIMIZE:
        ind = np.argmin(pool_obj, axis=1)
    if optmodel.modelSense == EPO.MAXIMIZE:
        ind = np.argmax(pool_obj, axis=1)
    sol = pool[ind]
    return sol
