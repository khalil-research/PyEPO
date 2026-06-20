"""Shared helpers for optimization model backends."""

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
