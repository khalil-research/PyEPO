#!/usr/bin/env python
# coding: utf-8
"""
Utility function
"""

import numpy as np

from pyepo import EPO
from pyepo.utlis import getArgs


def _solve_in_pass(cp, optmodel, processes, pool):
    """
    A function to solve optimization in the forward/backward pass
    """
    # number of instance
    ins_num = len(cp)
    # single-core
    if processes == 1:
        sol = []
        obj = []
        for i in range(ins_num):
            # solve
            optmodel.setObj(cp[i])
            solp, objp = optmodel.solve()
            sol.append(solp)
            obj.append(objp)
    # multi-core
    else:
        # get class
        model_type = type(optmodel)
        # get args
        args = getArgs(optmodel)
        # parallel computing
        res = pool.amap(_solveWithObj4Par, cp, [args] * ins_num,
                        [model_type] * ins_num).get()
        # get res
        sol = np.array(list(map(lambda x: x[0], res)))
        obj = np.array(list(map(lambda x: x[1], res)))
    return sol, obj


def _cache_in_pass(cp, optmodel, solpool):
    """
    A function to use solution pool in the forward/backward pass
    """
    # number of instance
    ins_num = len(cp)
    # best solution in pool
    solpool_obj = cp @ solpool.T
    if optmodel.modelSense == EPO.MINIMIZE:
        ind = np.argmin(solpool_obj, axis=1)
    if optmodel.modelSense == EPO.MAXIMIZE:
        ind = np.argmax(solpool_obj, axis=1)
    obj = np.take_along_axis(solpool_obj, ind.reshape(-1,1), axis=1).reshape(-1)
    sol = solpool[ind]
    return sol, obj


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
        if abs(z[i] - c[i] @ w[i]) / (abs(z[i]) + 1e-3) >= 1e-3:
            raise AssertionError(
                "Solution {} does not macth the objective value {}.".
                format(c[i] @ w[i], z[i][0]))


class sumGammaDistribution:
    """
    creates a generator of samples for the Sum-of-Gamma distribution
    """
    def __init__(self, kappa, n_iterations=10, seed=135):
        self.κ = kappa
        self.n_iterations = n_iterations
        self.rnd = np.random.RandomState(seed)

    def sample(self, size):
        # init samples
        samples = 0
        # calculate samples
        for i in range(1, self.n_iterations+1):
            samples += self.rnd.gamma(1/self.κ, self.κ/i, size)
        samples -= np.log(self.n_iterations)
        samples /= self.κ
        return samples
