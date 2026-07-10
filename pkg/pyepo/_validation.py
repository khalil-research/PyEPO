"""Package-wide validation for numeric configuration parameters."""

from __future__ import annotations

import math
from numbers import Real


def validate_positive(value: float, name: str) -> None:
    """Validate a finite, strictly positive real parameter."""
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number.")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number.")


def validate_positive_int(value: int, name: str) -> None:
    """Validate a strictly positive integer parameter."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def validate_nonnegative(value: float, name: str) -> None:
    """Validate a finite, non-negative real parameter."""
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number.")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")


def validate_probability(value: float, name: str) -> None:
    """Validate a finite real probability in the closed interval [0, 1]."""
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1].")


def validate_degree(deg: int) -> None:
    """Validate a positive integer polynomial degree."""
    validate_positive_int(deg, "deg")
