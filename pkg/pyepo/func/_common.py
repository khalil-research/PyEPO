"""Backend-independent policies shared by the Torch and JAX frontends."""

from pyepo import EPO


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
    """Validate a strictly positive numeric parameter."""
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def validate_positive_int(value, name: str) -> None:
    """Validate a strictly positive integer parameter."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def validate_nonnegative(value, name: str) -> None:
    """Validate a non-negative numeric parameter."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
