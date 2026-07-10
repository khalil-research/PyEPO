"""Framework-neutral runtime setup for differentiable optimization modules."""

from __future__ import annotations

import itertools
import multiprocessing as mp
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, cast

import numpy as np
from pathos.multiprocessing import ProcessingPool

from pyepo.func._common import is_minimize, validate_probability
from pyepo.func.utils import _close_pool, _init_worker_model
from pyepo.model.mpax import optMpaxModel
from pyepo.model.opt import optModel
from pyepo.utils import require_linear_objective

if TYPE_CHECKING:
    import logging

    from pyepo.data.dataset import optDataset

Reduction = Literal["mean", "sum", "none"]
T = TypeVar("T")


@dataclass(frozen=True)
class RuntimeState:
    """Validated state shared by the Torch and JAX frontends."""

    optmodel: optModel
    processes: int
    pool: ProcessingPool | None
    solve_ratio: float
    reduction: Reduction
    branch_rng: np.random.RandomState


def bind_runtime_state(owner: Any, runtime: RuntimeState) -> None:
    """Attach validated runtime state to a frontend module."""
    owner.optmodel = runtime.optmodel
    owner.processes = runtime.processes
    owner.pool = runtime.pool
    owner.solve_ratio = runtime.solve_ratio
    owner.reduction = runtime.reduction
    owner._branch_rng = runtime.branch_rng


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


# Unique pool ids and finalizers. The finalizer is a last-resort guard only;
# modules expose ``close`` so normal control flow releases workers explicitly.
_pool_ids = itertools.count()
_pool_finalizers: dict[int, weakref.finalize] = {}


def _finalize_pool(pool: ProcessingPool) -> None:
    """Close a pool from a weakref finalizer and drop its registration."""
    try:
        _close_pool(pool)
    finally:
        _pool_finalizers.pop(id(pool), None)


def close_solver_pool(pool: ProcessingPool | None) -> None:
    """Idempotently close a solver pool and detach its fallback finalizer."""
    if pool is None:
        return
    finalizer = _pool_finalizers.pop(id(pool), None)
    if finalizer is not None and finalizer.alive:
        finalizer()
    else:
        _close_pool(pool)


def close_runtime(owner: Any) -> None:
    """Release an owner's workers; later calls fall back to serial solving."""
    close_solver_pool(getattr(owner, "pool", None))
    owner.pool = None
    if getattr(owner, "processes", 1) > 1:
        owner.processes = 1


def create_solver_pool(
    optmodel: optModel,
    processes: int,
    *,
    owner=None,
    with_solver: bool = True,
) -> ProcessingPool | None:
    """Create a worker pool after validating its reconstruction recipe."""
    if processes == 1:
        return None
    # optional per-worker optmodel preload
    init_kwargs = {}
    if with_solver:
        spec = optmodel.to_spec()
        spec.validate_serializable()
        init_kwargs = {"initializer": _init_worker_model, "initargs": (spec,)}
    pool = ProcessingPool(processes, id=f"pyepo-{next(_pool_ids)}", **init_kwargs)
    if owner is not None:
        _pool_finalizers[id(pool)] = weakref.finalize(owner, _finalize_pool, pool)
    return pool


def init_solution_pool(
    dataset: optDataset | None,
    solve_ratio: float,
    require_solpool: bool,
    unique: Callable[[object], T],
) -> T | None:
    """Initialize a frontend-specific solution pool from an ``optDataset`` when needed."""
    if solve_ratio >= 1 and not require_solpool:
        return None

    from pyepo.data.dataset import optDataset

    if not isinstance(dataset, optDataset):
        raise TypeError("dataset is not an optDataset")
    return unique(dataset.sols)


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
    require_linear_objective(optmodel)
    is_minimize(optmodel.modelSense)
    validate_probability(solve_ratio, "solve_ratio")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"No reduction '{reduction}'.")

    normalized_processes = normalize_processes(optmodel, processes, logger)
    pool = create_solver_pool(
        optmodel,
        normalized_processes,
        owner=owner,
        with_solver=getattr(owner, "_pool_needs_optmodel", True),
    )
    logger.info("Num of cores: %d", normalized_processes)
    return RuntimeState(
        optmodel=optmodel,
        processes=normalized_processes,
        pool=pool,
        solve_ratio=solve_ratio,
        reduction=cast("Reduction", reduction),
        branch_rng=np.random.RandomState(seed),
    )
