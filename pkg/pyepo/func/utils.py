#!/usr/bin/env python
"""
Utility function
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from pyepo import EPO
from pyepo.model.mpax import optMpaxModel
from pyepo.utils import costToNumpy

if TYPE_CHECKING:
    from pyepo.func.abcmodule import optModule
    from pyepo.model.opt import optModel

logger = logging.getLogger(__name__)

# guards the per-batch MPAX device-mismatch warning so it fires once
_warned_mpax_device_mismatch = False


def _close_pool(pool) -> None:
    """Best-effort shutdown of a pathos ProcessingPool; swallows shutdown errors."""
    try:
        pool.close()
        pool.join()
        pool.clear()
    except Exception:  # noqa: BLE001  intentional best-effort shutdown
        pass


def _solve_or_cache(cp: torch.Tensor, module: optModule) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A function to get optimization solution in the forward/backward pass
    """
    optmodel = module.optmodel
    processes = module.processes
    pool = module.pool
    solpool = module.solpool
    if module._branch_rng.uniform() <= module.solve_ratio:
        sol, obj, solpool = _solve_in_pass(cp, optmodel, processes, pool, solpool)
    else:
        # cache branch implies solve_ratio < 1, so __init__ has populated solpool
        assert solpool is not None
        sol, obj, solpool = _cache_in_pass(cp, optmodel, solpool)
    module.solpool = solpool
    return sol, obj


def _solve_in_pass(
    cp: torch.Tensor,
    optmodel: optModel,
    processes: int,
    pool,
    solpool: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    A function to solve optimization and update solution pool
    """
    sol, obj = _solve_batch(cp, optmodel, processes, pool)
    # update solution pool on cp's device (tensor-side dedup, no CPU sync)
    if solpool is not None:
        solpool = _update_solution_pool(sol, solpool)
        if solpool.device != cp.device:
            solpool = solpool.to(cp.device)
    return sol, obj, solpool


def _solve_batch(
    cp: torch.Tensor | np.ndarray,
    optmodel: optModel,
    processes: int,
    pool,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        cp = optmodel.c  # pyright: ignore[reportAssignmentType]  # jax Array
        # batch solving
        sol, obj, _ = optmodel.batch_optimize(cp)
        # convert to torch
        sol = torch.from_dlpack(sol)
        obj = torch.from_dlpack(obj)
        if sol.device != device:
            # warn once; a persistent mismatch would otherwise flood logs per batch
            global _warned_mpax_device_mismatch
            if not _warned_mpax_device_mismatch:
                logger.warning(
                    "MPAX solutions on %s differ from input device %s; copying",
                    sol.device,
                    device,
                )
                _warned_mpax_device_mismatch = True
            sol, obj = sol.to(device), obj.to(device)
        # obj sense
        if optmodel.modelSense == EPO.MINIMIZE:
            pass
        elif optmodel.modelSense == EPO.MAXIMIZE:
            obj = -obj
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
    # single-core
    elif processes == 1:
        cp = costToNumpy(cp)
        sol_list: list = []
        obj_list: list = []
        for i in range(ins_num):
            optmodel.setObj(cp[i])
            solp, objp = optmodel.solve()
            sol_list.append(solp)
            obj_list.append(objp)
        # stack + dtype convert in a single call
        sol = torch.as_tensor(np.asarray(sol_list, dtype=np.float32)).to(device)
        obj = torch.tensor(obj_list, dtype=torch.float32, device=device)
    # multi-core (workers pre-loaded with optmodel via pool initializer)
    else:
        cp = costToNumpy(cp)
        res = pool.amap(_solveWithObj4Par, cp).get()
        # get res
        sol = torch.as_tensor(np.stack([r[0] for r in res]), dtype=torch.float32).to(device)
        obj = torch.tensor([r[1] for r in res], dtype=torch.float32, device=device)
    return sol, obj


def _update_solution_pool(
    sol: torch.Tensor,
    solpool: torch.Tensor | None,
) -> torch.Tensor:
    """
    Append rows of `sol` to `solpool` that aren't already present.

    Dedup runs with `torch.unique` + `torch.cdist` on `solpool`'s device.
    """
    if solpool is None:
        return torch.unique(sol, dim=0).clone()
    if sol.device != solpool.device:
        sol = sol.to(solpool.device)
    sol_uniq = torch.unique(sol, dim=0)
    # exact-equality via L1 distance (== 0 ⇒ identical row)
    dists = torch.cdist(sol_uniq, solpool, p=1.0)
    is_new = (dists != 0).all(dim=1)
    if bool(is_new.any()):
        solpool = torch.cat((solpool, sol_uniq[is_new]), dim=0)
    return solpool


def _cache_in_pass(
    cp: torch.Tensor,
    optmodel: optModel,
    solpool: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A function to use solution pool in the forward/backward pass
    """
    # move solpool to the correct device
    if solpool.device != cp.device:
        solpool = solpool.to(cp.device)
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
    return sol, obj, solpool


# per-worker optmodel, built by `_init_worker_model` at pool startup
_worker_model = None


def _init_worker_model(model_type: type, args: dict) -> None:
    """Pool-initializer hook: build the optmodel once per worker process."""
    global _worker_model
    _worker_model = model_type(**args)


def _solveWithObj4Par(cost: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve a single instance in a pool worker using the pre-built optmodel."""
    assert _worker_model is not None, "_init_worker_model must run before this"
    _worker_model.setObj(cost)
    sol, obj = _worker_model.solve()
    return np.asarray(sol, dtype=np.float32), obj


def _check_sol(c: torch.Tensor, w: torch.Tensor, z: torch.Tensor) -> None:
    """
    A function to check solution is correct
    """
    z_flat = z.squeeze(-1) if z.dim() > 1 else z
    error = torch.abs(z_flat - torch.einsum("bi,bi->b", c, w)) / (torch.abs(z_flat) + 1e-3)
    if torch.any(error >= 1e-3):
        raise AssertionError("Some solutions do not match the objective value.")


def _mask_pred(noises: torch.Tensor, optmodel: optModel) -> torch.Tensor:
    """
    Zero a cost perturbation outside the predicted positions.

    No-op when every variable carries a predicted cost (``c_pred_index`` is
    ``None``); under partial prediction the known fixed costs are left
    unperturbed.
    """
    idx = optmodel.c_pred_index
    if idx is None:
        return noises
    mask = noises.new_zeros(noises.shape[-1])
    mask[torch.as_tensor(idx, dtype=torch.long, device=noises.device)] = 1.0
    return noises * mask


def _torch_generator(
    cache: dict[str, torch.Generator],
    device: torch.device | str,
    seed: int,
) -> torch.Generator:
    """Lazily create and cache a per-device torch.Generator seeded from `seed`."""
    dev = torch.device(device) if isinstance(device, str) else device
    key = str(dev)
    gen = cache.get(key)
    if gen is None:
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
        cache[key] = gen
    return gen


class sumGammaDistribution:
    """
    creates a generator of samples for the Sum-of-Gamma distribution
    """

    def __init__(self, kappa: float, n_iterations: int = 10, seed: int = 135) -> None:
        self.κ = kappa
        self.n_iterations = n_iterations
        self.seed = seed
        self.rnd = np.random.RandomState(seed)
        self._gen_cache: dict[str, torch.Generator] = {}

    def sample(
        self,
        size: int | tuple[int, ...],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> np.ndarray | torch.Tensor:
        # numpy path
        if device is None:
            samples = 0
            for i in range(1, self.n_iterations + 1):
                samples += self.rnd.gamma(1 / self.κ, self.κ / i, size)
            samples -= np.log(self.n_iterations)
            samples /= self.κ
            return samples
        # torch path
        size_t = size if isinstance(size, tuple) else (size,)
        gen = _torch_generator(self._gen_cache, device, self.seed)
        # alpha block: n_iterations on the leading axis for a single sampler call
        alpha = torch.full((self.n_iterations, *size_t), 1.0 / self.κ, device=device, dtype=dtype)
        gammas = torch._standard_gamma(alpha, generator=gen)
        # per-iteration weights kappa/i, broadcast over the trailing sample axes
        weights = self.κ / torch.arange(1, self.n_iterations + 1, device=device, dtype=dtype)
        weights = weights.view(self.n_iterations, *([1] * len(size_t)))
        samples = (gammas * weights).sum(dim=0)
        samples.sub_(math.log(self.n_iterations))
        samples.div_(self.κ)
        return samples
