"""Shared validation for synthetic data generators."""

import math
from numbers import Real


def validate_degree(deg: int) -> None:
    """Validate a positive integer polynomial degree."""
    if not isinstance(deg, int) or isinstance(deg, bool) or deg <= 0:
        raise ValueError(f"deg = {deg} should be a positive integer.")


def validate_nonnegative(value: float, name: str) -> None:
    """Validate a finite, non-negative real generator parameter."""
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} = {value} should be a finite non-negative number.")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} = {value} should be a finite non-negative number.")


def validate_probability(value: float, name: str) -> None:
    """Validate a finite real value in the closed interval [0, 1]."""
    validate_nonnegative(value, name)
    if float(value) > 1:
        raise ValueError(f"{name} = {value} should be in [0, 1].")


def validate_positive_int(value: int, name: str) -> None:
    """Validate a strictly positive integer parameter."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} = {value} should be a positive integer.")
