"""Framework-neutral runtime setup for differentiable optimization modules."""

from __future__ import annotations

import multiprocessing as mp
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from pathos.multiprocessing import ProcessingPool

from pyepo import EPO
from pyepo.func.utils import _close_pool, _init_worker_model
from pyepo.model.mpax import optMpaxModel
from pyepo.model.opt import optModel

if TYPE_CHECKING:
    import logging

Reduction = Literal["mean", "sum", "none"]


@dataclass(frozen=True)
class RuntimeState:
    """Validated state shared by the Torch and JAX frontends."""

    optmodel: optModel
    processes: int
    pool: ProcessingPool | None
    solve_ratio: float
    reduction: Reduction
    branch_rng: np.random.RandomState


def normalize_processes(
    optmodel: optModel,
    processes: int,
    logger: logging.Logger,
) -> int:
    """Validate and normalize a requested solver-process count."""
    cpu_count = mp.cpu_count()
    if processes < 0:
        raise ValueError(f"Invalid processors number {processes}, only {cpu_count} cores.")
    if isinstance(optmodel, optMpaxModel) and processes != 1:
        logger.warning("MPAX does not support multiprocessing. Setting `processes = 1`.")
        return 1
    if processes > cpu_count:
        raise ValueError(f"Invalid processors number {processes}, only {cpu_count} cores.")
    return cpu_count if processes == 0 else processes


def create_solver_pool(
    optmodel: optModel,
    processes: int,
    *,
    owner=None,
) -> ProcessingPool | None:
    """Create a worker pool, optionally tied to an owner's lifetime."""
    if processes == 1:
        return None
    pool = ProcessingPool(
        processes,
        initializer=_init_worker_model,
        initargs=(optmodel.to_spec(),),
    )
    if owner is not None:
        weakref.finalize(owner, _close_pool, pool)
    return pool


def init_runtime(
    owner,
    optmodel: optModel,
    processes: int,
    solve_ratio: float,
    reduction: str,
    seed: int | None,
    logger: logging.Logger,
) -> RuntimeState:
    """Validate common module arguments and initialize solver runtime state."""
    if not isinstance(optmodel, optModel):
        raise TypeError("arg model is not an optModel")
    if optmodel.modelSense not in (EPO.MINIMIZE, EPO.MAXIMIZE):
        raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
    if not 0 <= solve_ratio <= 1:
        raise ValueError(f"Invalid solving ratio {solve_ratio}. It should be between 0 and 1.")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"No reduction '{reduction}'.")

    normalized_processes = normalize_processes(optmodel, processes, logger)
    pool = create_solver_pool(optmodel, normalized_processes, owner=owner)
    logger.info("Num of cores: %d", normalized_processes)
    return RuntimeState(
        optmodel=optmodel,
        processes=normalized_processes,
        pool=pool,
        solve_ratio=solve_ratio,
        reduction=cast("Reduction", reduction),
        branch_rng=np.random.RandomState(seed),
    )
