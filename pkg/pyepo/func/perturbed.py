#!/usr/bin/env python
# coding: utf-8
"""
Perturbed optimization function
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

class perturbedOpt(nn.Module):
    """
    A autograd module for perturbed optimizer, models of random optimizers with
    perturbed inputs.

    For the perturbed optimizer, the cost vector need to be predicted from
    contextual data and are perturbed with Gaussian noise.

    The perturbed optimizer differentiable in its inputs with non-zero Jacobian.
    Thus, allows us to design an algorithm based on stochastic gradient descent.
    """

    def __init__(self, optmodel, n_samples=10, epsilon=1.0, processes=1):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            epsilon (float): the amplitude of the perturbation
            processes (int): number of processors, 1 for single-core, 0 for all of cores
        """
        super().__init__()
        # optimization model
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel")
        self.optmodel = optmodel
        # number of samples
        self.n_samples = n_samples
        # perturbation amplitude
        self.epsilon = epsilon
        # num of processors
        if processes not in range(mp.cpu_count()+1):
            raise ValueError("Invalid processors number {}, only {} cores.".
                format(processes, mp.cpu_count()))
        self.processes = processes
        print("Num of cores: {}".format(self.processes))
        # build optimizer
        self.ptb = perturbedOptFunc()

    def forward(self, pred_cost):
        """
        Forward pass
        """
        sols = self.ptb.apply(pred_cost, self.optmodel, self.n_samples,
                              self.epsilon, self.processes)
        return sols


class perturbedOptFunc(Function):
    """
    A autograd function for perturbed optimizer
    """

    @staticmethod
    def forward(ctx, pred_cost, optmodel, n_samples, epsilon, processes):
        """
        Forward pass for DBB

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            optmodel (optModel): an PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            epsilon (float): the amplitude of the perturbation
            processes (int): number of processors, 1 for single-core, 0 for all of cores

        Returns:
            torch.tensor: predicted solutions
        """
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach().to("cpu").numpy()
        # sample perturbations
        noises = np.random.normal(0, 1, size=(n_samples, *cp.shape))
        ptb_c = cp + epsilon * noises
        # solve with perturbation
        ptb_sols = _solve_in_forward(ptb_c, optmodel, processes)
        # expectation
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
        ctx.epsilon = epsilon
        return e_sol

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for DBB
        """
        ptb_sols, noises = ctx.saved_tensors
        optmodel = ctx.optmodel
        n_samples = ctx.n_samples
        epsilon = ctx.epsilon
        grad = torch.einsum("nbd,bn->bd",
                            noises,
                            torch.einsum("bnd,bd->bn", ptb_sols, grad_output))
        grad /= n_samples * epsilon
        return grad, None, None, None, None


def _solve_in_forward(ptb_c, optmodel, processes):
    """
    A function to solve optimization in the forward pass
    """
    # number of instance
    n_samples, ins_num, _ = ptb_c.shape
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
        # number of processes
        processes = mp.cpu_count() if not processes else processes
        # get class
        model_type = type(optmodel)
        # get args
        args = getArgs(optmodel)
        # parallel computing
        with ProcessingPool(processes) as pool:
            ptb_sols = pool.amap(_solveWithObj4Par, ptb_c.transpose(1,0,2),
                                 [args] * ins_num, [model_type] * ins_num).get()
    return np.array(ptb_sols)


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
