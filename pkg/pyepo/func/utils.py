#!/usr/bin/env python
# coding: utf-8
"""
Utility function
"""

import numpy as np
import torch

from pyepo import EPO
from pyepo.utils import getArgs
from pyepo.model.mpax import optMpaxModel

try:
    import jax
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
        # move solpool to the correct device
        if module.solpool.device != cp.device:
            module.solpool = module.solpool.to(cp.device)
        sol, obj = _cache_in_pass(cp, module.optmodel, module.solpool)
    return sol, obj


def _solve_in_pass(cp, optmodel, processes, pool):
    """
    A function to solve optimization in the forward/backward pass
    """
    # get device
    device = cp.device if isinstance(cp, torch.Tensor) else torch.device("cpu")
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
            pass
        elif optmodel.modelSense == EPO.MAXIMIZE:
            obj = - obj
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
    # single-core
    elif processes == 1:
        cp = cp.detach().cpu().numpy() if isinstance(cp, torch.Tensor) else np.asarray(cp)
        sol = []
        obj = []
        for i in range(ins_num):
            # solve
            optmodel.setObj(cp[i])
            solp, objp = optmodel.solve()
            sol.append(np.asarray(solp, dtype=np.float32))
            obj.append(objp)
        # to tensor
        sol = torch.as_tensor(np.stack(sol), dtype=torch.float32).to(device)
        obj = torch.tensor(obj, dtype=torch.float32, device=device)
    # multi-core
    else:
        cp = cp.detach().cpu().numpy() if isinstance(cp, torch.Tensor) else np.asarray(cp)
        # get class
        model_type = type(optmodel)
        # get args
        args = getArgs(optmodel)
        # parallel computing
        res = pool.amap(_solveWithObj4Par, cp, [args] * ins_num,
                        [model_type] * ins_num).get()
        # get res
        sol = torch.as_tensor(np.stack([r[0] for r in res]), dtype=torch.float32).to(device)
        obj = torch.tensor([r[1] for r in res], dtype=torch.float32, device=device)
    return sol, obj


def _cache_in_pass(cp, optmodel, solpool):
    """
    A function to use solution pool in the forward/backward pass
    """
    # get device
    device = cp.device
    # best solution in pool
    solpool_obj = torch.matmul(cp, solpool.T)
    if optmodel.modelSense == EPO.MINIMIZE:
        ind = torch.argmin(solpool_obj, dim=1)
    elif optmodel.modelSense == EPO.MAXIMIZE:
        ind = torch.argmax(solpool_obj, dim=1)
    else:
        raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
    obj = solpool_obj.gather(1, ind.view(-1, 1)).squeeze(1)
    sol = solpool[ind]
    return sol, obj


# worker-local model cache (persists across calls in pathos worker processes)
_worker_model = None
_worker_model_key = None

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
    global _worker_model, _worker_model_key
    # reuse cached model if same type; rebuild only on first call or type change
    key = model_type.__qualname__
    if _worker_model_key != key:
        _worker_model = model_type(**args)
        _worker_model_key = key
    optmodel = _worker_model
    # set obj
    optmodel.setObj(cost)
    # solve
    sol, obj = optmodel.solve()
    sol = np.asarray(sol, dtype=np.float32)
    return sol, obj


def _check_sol(c, w, z):
    """
    A function to check solution is correct
    """
    z_flat = z.squeeze(-1) if z.dim() > 1 else z
    error = torch.abs(z_flat - torch.einsum("bi,bi->b", c, w)) / (torch.abs(z_flat) + 1e-3)
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
