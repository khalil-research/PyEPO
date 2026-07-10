"""Backend-independent policies shared by the Torch and JAX frontends."""

from typing import Optional, TypeVar

from pyepo import EPO
from pyepo._validation import (
    validate_nonnegative,
    validate_positive,
    validate_positive_int,
    validate_probability,
)

T = TypeVar("T")

__all__ = [
    "is_minimize",
    "require_solution_pool",
    "solution_pool_tolerance",
    "validate_nonnegative",
    "validate_positive",
    "validate_positive_int",
    "validate_probability",
]


def is_minimize(model_sense) -> bool:
    """Return the objective direction, rejecting unsupported sense values."""
    if model_sense == EPO.MINIMIZE:
        return True
    if model_sense == EPO.MAXIMIZE:
        return False
    raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")


def solution_pool_tolerance(num_cost: int) -> float:
    """L1 tolerance used to deduplicate approximate solver solutions."""
    return min(1e-4 * num_cost, 0.1)


def require_solution_pool(solpool: Optional[T]) -> T:
    """Return an initialized solution pool or raise a stable runtime error."""
    if solpool is None:
        raise RuntimeError(
            "Solution pool is unavailable; provide an optDataset when pool-based solving is enabled."
        )
    return solpool
