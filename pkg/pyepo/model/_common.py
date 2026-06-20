"""Shared helpers for optimization model backends."""

import math
from numbers import Real

import numpy as np


def validate_objective_shape(
    cost,
    expected_size: int | tuple[int, ...],
    *,
    allow_batch: bool = False,
    full: bool = False,
) -> None:
    """Validate objective dimensionality without converting its array backend."""
    shape = getattr(cost, "shape", None)
    if shape is None:
        shape = np.asarray(cost).shape
    sizes = (expected_size,) if isinstance(expected_size, int) else expected_size
    valid_ndim = len(shape) in (1, 2) if allow_batch else len(shape) == 1
    if not valid_ndim or shape[-1] not in sizes:
        target = "variables" if full else "cost variables"
        dimensions = ", ".join(str(size) for size in sizes)
        batch_suffix = " or a two-dimensional batch" if allow_batch else ""
        raise ValueError(
            f"Objective must be a one-dimensional vector{batch_suffix} whose last "
            f"dimension matches the number of {target} ({dimensions})."
        )


def validate_constraint(
    coefs,
    rhs,
    expected_size: int,
    *,
    full: bool = False,
) -> float:
    """Validate a finite linear cut and return its scalar right-hand side."""
    validate_objective_shape(coefs, expected_size, full=full)
    try:
        array = np.asarray(coefs)
    except (TypeError, ValueError) as exc:
        raise ValueError("Constraint coefficients must be NumPy-compatible.") from exc
    valid_type = np.issubdtype(array.dtype, np.number) and not np.issubdtype(
        array.dtype, np.complexfloating
    )
    finite = np.isfinite(array).all() if valid_type else False
    if not valid_type:
        raise ValueError("Constraint coefficients must be real numbers.")
    if not finite:
        raise ValueError("Constraint coefficients must contain only finite values.")
    if not isinstance(rhs, Real) or isinstance(rhs, bool):
        raise ValueError("Constraint rhs must be a finite number.")
    scalar_rhs = float(rhs)
    if not math.isfinite(scalar_rhs):
        raise ValueError("Constraint rhs must be a finite number.")
    return scalar_rhs
