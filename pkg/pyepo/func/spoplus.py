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
from torch import nn

from pyepo import EPO
from pyepo.data.dataset import optDataset
from pyepo.model.opt import optModel
from pyepo.utlis import getArgs


class SPOPlus(nn.Module):
    """
    A autograd module for SPO+ Loss, as a surrogate loss function of SPO Loss,
    which measures the decision error of optimization problem.

    For SPO/SPO+ Loss, the objective function is linear and constraints are
    known and fixed, but the cost vector need to be predicted from contextual
    data.

    The SPO+ Loss is convex with subgradient. Thus, allows us to design an
    algorithm based on stochastic gradient descent.
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__()
        # optimization model
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel")
        self.optmodel = optmodel
        # number of processes
        if processes not in range(mp.cpu_count()+1):
            raise ValueError("Invalid processors number {}, only {} cores.".
                format(processes, mp.cpu_count()))
        self.processes = mp.cpu_count() if not processes else processes
        # single-core
        if processes == 1:
            self.pool = None
        # multi-core
        else:
            self.pool = ProcessingPool(processes)
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
        self.spop = SPOPlusFunc()

    def forward(self, pred_cost, true_cost, true_sol, true_obj, reduction="mean"):
        """
        Forward pass
        """
        loss = self.spop.apply(pred_cost, true_cost, true_sol, true_obj,
                               self.optmodel, self.processes, self.pool,
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



class SPOPlusFunc(Function):
    """
    A autograd function for SPO+ Loss
    """

    @staticmethod
    def forward(ctx, pred_cost, true_cost, true_sol, true_obj,
                optmodel, processes, pool, solve_ratio, module):
        """
        Forward pass for SPO+

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_cost (torch.tensor): a batch of true values of the cost
            true_sol (torch.tensor): a batch of true optimal solutions
            true_obj (torch.tensor): a batch of true optimal objective values
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            pool (ProcessPool): process pool object
            solve_ratio (float): the ratio of new solutions computed during training
            module (nn.Module): SPOPlus modeul

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
        if np.random.uniform() <= solve_ratio:
            sol, loss = _solve_in_forward(cp, c, w, z, optmodel, processes, pool)
            if solve_ratio < 1:
                module.solpool = np.concatenate((module.solpool, sol))
        else:
            sol, loss = _cache_in_forward(cp, c, w, z, optmodel, module.solpool)
        # sense
        if optmodel.modelSense == EPO.MINIMIZE:
            loss = np.array(loss)
        if optmodel.modelSense == EPO.MAXIMIZE:
            loss = - np.array(loss)
        # convert to tensor
        loss = torch.FloatTensor(loss).to(device)
        sol = np.array(sol)
        sol = torch.FloatTensor(sol).to(device)
        # save solutions
        ctx.save_for_backward(true_sol, sol)
        # add other objects to ctx
        ctx.optmodel = optmodel
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
        return grad_output * grad, None, None, None, None, None, None, None, None


def _solve_in_forward(cp, c, w, z, optmodel, processes, pool):
    """
    A function to solve optimization in the forward pass
    """
    # number of instance
    ins_num = len(z)
    # single-core
    if processes == 1:
        loss = []
        sol = []
        for i in range(ins_num):
            # solve
            optmodel.setObj(2 * cp[i] - c[i])
            solq, objq = optmodel.solve()
            # calculate loss
            loss.append(-objq + 2 * np.dot(cp[i], w[i]) - z[i])
            # solution
            sol.append(solq)
    # multi-core
    else:
        # get class
        model_type = type(optmodel)
        # get args
        args = getArgs(optmodel)
        res = pool.amap(
              _solveWithObj4Par,
              2 * cp - c,
              [args] * ins_num,
              [model_type] * ins_num).get()
        # get res
        sol = np.array(list(map(lambda x: x[0], res)))
        obj = np.array(list(map(lambda x: x[1], res)))
        # calculate loss
        loss = []
        for i in range(ins_num):
            loss.append(- obj[i] + 2 * np.dot(cp[i], w[i]) - z[i])
    return sol, loss


def _cache_in_forward(cp, c, w, z, optmodel, solpool):
    """
    A function to use solution pool in the forward pass
    """
    # number of instance
    ins_num = len(z)
    # best solution in pool
    solpool_obj = (2 * cp - c) @ solpool.T
    if optmodel.modelSense == EPO.MINIMIZE:
        ind = np.argmin(solpool_obj, axis=1)
    if optmodel.modelSense == EPO.MAXIMIZE:
        ind = np.argmax(solpool_obj, axis=1)
    obj = np.take_along_axis(solpool_obj, ind.reshape(-1,1), axis=1).reshape(-1)
    sol = solpool[ind]
    # calculate loss
    loss = []
    for i in range(ins_num):
        loss.append(- obj[i] + 2 * np.dot(cp[i], w[i]) - z[i])
    return sol, loss


def _solveWithObj4Par(cost, args, model_type):
    """
    A function to solve function in parallel processors

    Args:
        cost (np.ndarray): cost of objective function
        args (dict): optModel args
        model_type (ABCMeta): optModel class type

    Returns:
        tuple: optimal solution (list) and objective value (float)
    """
    # rebuild model
    optmodel = model_type(**args)
    # set obj
    optmodel.setObj(cost)
    # solve
    sol, obj = optmodel.solve()
    return sol, obj


def _check_sol(c, w, z):
    """
    A function to check solution is correct
    """
    ins_num = len(z)
    for i in range(ins_num):
        if abs(z[i] - np.dot(c[i], w[i])) / (abs(z[i]) + 1e-3) >= 1e-3:
            raise AssertionError(
                "Solution {} does not macth the objective value {}.".
                format(np.dot(c[i], w[i]), z[i][0]))
