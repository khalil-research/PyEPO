"""Backend-independent policies shared by the Torch and JAX frontends."""

import math
from numbers import Real
from typing import Optional, TypeVar

from pyepo import EPO

T = TypeVar("T")


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


def validate_positive(value, name: str) -> None:
    """Validate a finite, strictly positive real parameter."""
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number.")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number.")


def validate_positive_int(value, name: str) -> None:
    """Validate a strictly positive integer parameter."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def validate_nonnegative(value, name: str) -> None:
    """Validate a finite, non-negative real parameter."""
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number.")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")


def validate_probability(value, name: str) -> None:
    """Validate a finite real probability in the closed interval [0, 1]."""
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1].")


def require_solution_pool(solpool: Optional[T]) -> T:
    """Return an initialized solution pool or raise a stable runtime error."""
    if solpool is None:
        raise RuntimeError(
            "Solution pool is unavailable; provide an optDataset when pool-based solving is enabled."
        )
    return solpool
