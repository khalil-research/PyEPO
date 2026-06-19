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
