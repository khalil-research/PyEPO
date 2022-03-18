#!/usr/bin/env python
# coding: utf-8
"""
Diffenretiable Black-box optimization function
"""

import multiprocessing as mp

import numpy as np
import torch
from pathos.multiprocessing import ProcessingPool
from torch.autograd import Function

from pyepo import EPO
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
    A autograd function for diffenretiable black-box optimizer, which yield
    optimal a solution and derive a gradient.

    For diffenretiable block-box, the objective function is linear and
    constraints are known and fixed, but the cost vector need to be predicted
    from contextual data.

    The block-box approximate gradient of optimizer smoothly. Thus, allows us to
    design an algorithm based on stochastic gradient descent.
    """

    def __init__(self, optmodel, lambd=10, processes=1):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            lambd (float): a hyperparameter for diffenretiable block-box to contral interpolation degree
            processes (int): number of processors, 1 for single-core, 0 for all of cores
        """
        super().__init__()
        # optimization model
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel.")
        global _SPO_FUNC_BB_OPTMODEL
        _SPO_FUNC_BB_OPTMODEL = optmodel
        # smoothing parameter
        if lambd <= 0:
            raise ValueError("lambda is not positive.")
        global _SPO_FUNC_BB_LAMBDA
        _SPO_FUNC_BB_LAMBDA = lambd
        # num of processors
        if processes not in range(mp.cpu_count()+1):
            raise ValueError("Invalid processors number {}, only {} cores.".
                format(processes, mp.cpu_count()))
        global _SPO_FUNC_BB_PROCESSES
        _SPO_FUNC_BB_PROCESSES = processes
        print("Num of cores: {}".format(_SPO_FUNC_BB_PROCESSES))

    @staticmethod
    def forward(ctx, pred_cost):
        """
        Forward pass in neural network.

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost

        Returns:
            torch.tensor: predicted solutions
        """
        ins_num = len(pred_cost)
        # get device
        device = pred_cost.device
        # get global
        optmodel = _SPO_FUNC_BB_OPTMODEL
        processes = _SPO_FUNC_BB_PROCESSES
        # convert tenstor
        cp = pred_cost.to("cpu").numpy()
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
        # convert to tensor
        sol = np.array(sol)
        pred_sol = torch.FloatTensor(sol).to(device)
        # save
        ctx.save_for_backward(pred_cost, pred_sol)
        return pred_sol

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass in neural network
        """
        pred_cost, pred_sol = ctx.saved_tensors
        ins_num = len(pred_cost)
        # get device
        device = pred_cost.device
        # get global
        optmodel = _SPO_FUNC_BB_OPTMODEL
        lambd = _SPO_FUNC_BB_LAMBDA
        processes = _SPO_FUNC_BB_PROCESSES
        # convert tenstor
        cp = pred_cost.to("cpu").numpy()
        wp = pred_sol.to("cpu").numpy()
        dl = grad_output.to("cpu").numpy()
        # perturbed costs
        cq = cp + lambd * dl
        # single-core
        if processes == 1:
            grad = []
            for i in range(len(cp)):
                # solve
                optmodel.setObj(cq[i])
                solq, _ = optmodel.solve()
                # gradient of continuous interpolation
                grad.append((solq - wp[i]) / lambd)
        # multi-core
        else:
            # get class
            model_type = type(optmodel)
            # get args
            args = getArgs(optmodel)
            # number of processes
            processes = mp.cpu_count() if not processes else processes
            # parallel computing
            with ProcessingPool(processes) as pool:
                sol = pool.amap(solveWithObj4Par, cq, [args] * ins_num,
                                [model_type] * ins_num).get()
            # get gradient
            grad = []
            for i in range(ins_num):
                grad.append((sol[i] - wp[i]) / lambd)
        # convert to tensor
        grad = np.array(grad)
        grad = torch.FloatTensor(grad).to(device)
        return grad
