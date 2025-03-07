#!/usr/bin/env python
# coding: utf-8
"""
Utility function
"""

import numpy as np
import torch

from pyepo import EPO
from pyepo.utlis import getArgs
from pyepo.model.mpax import optMpaxModel

try:
    import jax
    from jax import numpy as jnp
    from mpax import create_lp, r2HPDHG
except ImportError:
    pass


def _solve_or_cache(cp, module):
    """
    A function to get optimization solution in the forward/backward pass
    """
    # solve optimization
    if np.random.uniform() <= module.solve_ratio:
        sol, obj = _solve_in_pass(cp, module.optmodel, module.processes, module.pool)
        if module.solve_ratio < 1:
            # add into solpool
            module._update_solution_pool(sol)
    # best cached solution
    else:
        sol, obj = _cache_in_pass(cp, module.optmodel, module.solpool)
    return sol, obj


def _solve_in_pass(cp, optmodel, processes, pool):
    """
    A function to solve optimization in the forward/backward pass
    """
    # get device
    device = cp.device
    # number of instance
    ins_num = len(cp)
    # MPAX batch solving
    if isinstance(optmodel, optMpaxModel):
        # get params
        optmodel.setObj(cp)
        cp = optmodel.c
        # batch solving
        sol, obj = optmodel.batch_optimize(cp)
        # convert to torch
        sol = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(sol)).to(device)
        obj = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(obj)).to(device)
        # obj sense
        if optmodel.modelSense == EPO.MINIMIZE:
            obj = obj
        elif optmodel.modelSense == EPO.MAXIMIZE:
            obj = - obj
        else:
            raise ValueError("Invalid modelSense.")
    # single-core
    elif processes == 1:
        sol = []
        obj = []
        for i in range(ins_num):
            # solve
            optmodel.setObj(cp[i])
            solp, objp = optmodel.solve()
            sol.append(torch.as_tensor(solp))
            obj.append(objp)
        # to tensor
        sol = torch.stack(sol, dim=0).to(device)
        obj = torch.tensor(obj, dtype=torch.float32, device=device)
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
        sol = torch.stack([r[0] for r in res], dim=0).to(device)
        obj = torch.tensor([r[1] for r in res], dtype=torch.float32, device=device)
    return sol, obj


def _cache_in_pass(cp, optmodel, solpool):
    """
    A function to use solution pool in the forward/backward pass
    """
    # get device
    device = cp.device
    # solpool is on the same device
    if solpool.device != device:
        solpool = solpool.to(device)
    # best solution in pool
    solpool_obj = torch.matmul(cp, solpool.T)
    if optmodel.modelSense == EPO.MINIMIZE:
        ind = torch.argmin(solpool_obj, dim=1)
    if optmodel.modelSense == EPO.MAXIMIZE:
        ind = torch.argmax(solpool_obj, dim=1)
    obj = solpool_obj.gather(1, ind.view(-1, 1)).squeeze(1)
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
    # to tensor
    sol = torch.tensor(sol, dtype=torch.float32)
    return sol, obj


def _check_sol(c, w, z):
    """
    A function to check solution is correct
    """
    error = torch.abs(z - torch.einsum("bi,bi->b", c, w)) / (torch.abs(z) + 1e-3)
    if torch.any(error >= 1e-3):
        raise AssertionError("Some solutions do not match the objective value.")


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
