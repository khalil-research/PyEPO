"""Validation helpers for metric configuration."""

import math
from numbers import Real


def validate_tolerance(tolerance: float) -> None:
    """Validate a finite, strictly positive numerical tolerance."""
    if not isinstance(tolerance, Real) or isinstance(tolerance, bool):
        raise ValueError("tolerance must be a finite positive number.")
    number = float(tolerance)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("tolerance must be a finite positive number.")


def validate_retry_count(max_iter: int) -> None:
    """Validate the recursive retry budget while preserving exhaustion semantics."""
    if not isinstance(max_iter, int) or isinstance(max_iter, bool):
        raise ValueError("max_iter must be a positive integer.")
    if max_iter <= 0:
        raise RuntimeError("Max iterations reached in calUnambRegret.")
