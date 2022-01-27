#!/usr/bin/env python
# coding: utf-8
"""
SPO+ Loss function
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
        tuple: optimal solution (list) and objective value (float)
    """
    # rebuild model
    model = model_type(**args)
    # set obj
    model.setObj(cost)
    # solve
    sol, obj = model.solve()
    return sol, obj


class SPOPlus(Function):
    """
    SPO+ Loss function, a surrogate loss function of SPO Loss, which measures
    the decision error (optimality gap) of optimization problem.

    For SPO/SPO+ Loss, the objective function is linear and constraints are
    known and fixed, but the cost vector need to be predicted from contextual
    data.

    The SPO+ Loss is convex with subgradient. Thus, allows us to design an
    algorithm based on stochastic gradient descent.
    """

    def __init__(self, model, processes=1):
        """
        Args:
            model (optModel): optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
        """
        super().__init__()
        # optimization model
        if not isinstance(model, optModel):
            raise TypeError("arg model is not an optModel")
        global _SPO_FUNC_SPOP_OPTMODEL
        _SPO_FUNC_SPOP_OPTMODEL = model
        # num of processors
        if processes not in range(mp.cpu_count()+1):
            raise ValueError("Invalid processors number {}, only {} cores.".
                format(processes, mp.cpu_count()))
        global _SPO_FUNC_SPOP_PROCESSES
        _SPO_FUNC_SPOP_PROCESSES = processes
        print("Num of cores: {}".format(_SPO_FUNC_SPOP_PROCESSES))

    @staticmethod
    def forward(ctx, pred_cost, true_cost, true_sol, true_obj):
        """
        Forward pass in neural network

        Args:
            pred_cost (torch.tensor): predicted costs
            true_cost (torch.tensor): true costs
            true_sol (torch.tensor): true solutions
            true_obj (torch.tensor): true objective values

        Returns:
            torch.tensor: SPO+ loss
        """
        # get device
        device = pred_cost.device
        # get global
        model = _SPO_FUNC_SPOP_OPTMODEL
        processes = _SPO_FUNC_SPOP_PROCESSES
        # convert tenstor
        cp = pred_cost.to("cpu").numpy()
        c = true_cost.to("cpu").numpy()
        w = true_sol.to("cpu").numpy()
        z = true_obj.to("cpu").numpy()
        # check sol
        ins_num = len(z)
        for i in range(ins_num):
            if abs(z[i] - np.dot(c[i], w[i])) / (abs(z[i]) + 1e-3) >= 1e-3:
                raise AssertionError(
                    "Solution {} does not macth the objective value {}.".
                    format(np.dot(c[i], w[i]), z[i][0]))
        # single-core
        if processes == 1:
            loss = []
            sol = []
            for i in range(ins_num):
                # solve
                model.setObj(2 * cp[i] - c[i])
                solq, objq = model.solve()
                # calculate loss
                loss.append(-objq + 2 * np.dot(cp[i], w[i]) - z[i])
                # solution
                sol.append(solq)
        # multi-core
        else:
            # get class
            model_type = type(model)
            # get args
            args = getArgs(model)
            # number of processes
            processes = mp.cpu_count() if not processes else processes
            # parallel computing
            with ProcessingPool(processes) as pool:
                res = pool.amap(
                    solveWithObj4Par,
                    2 * cp - c,
                    [args] * ins_num,
                    [model_type] * ins_num,
                ).get()
            # get res
            sol = np.array(list(map(lambda x: x[0], res)))
            obj = np.array(list(map(lambda x: x[1], res)))
            # calculate loss
            loss = []
            for i in range(ins_num):
                loss.append(- obj[i] + 2 * np.dot(cp[i], w[i]) - z[i])
        # sense
        if model.modelSense == EPO.MINIMIZE:
            loss = np.array(loss)
        if model.modelSense == EPO.MAXIMIZE:
            loss = - np.array(loss)
        # convert to tensor
        loss = torch.FloatTensor(loss).to(device)
        sol = np.array(sol)
        sol = torch.FloatTensor(sol).to(device)
        # save solutions
        ctx.save_for_backward(true_sol, sol)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass in neural network
        """
        w, wq = ctx.saved_tensors
        model = _SPO_FUNC_SPOP_OPTMODEL
        if model.modelSense == EPO.MINIMIZE:
            grad = 2 * (w - wq).mean(0)
        if model.modelSense == EPO.MAXIMIZE:
            grad = 2 * (wq - w).mean(0)
        return grad_output * grad, None, None, None
