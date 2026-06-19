"""Shared validation for synthetic data generators."""


def validate_degree(deg: int) -> None:
    """Validate a positive integer polynomial degree."""
    if not isinstance(deg, int):
        raise ValueError(f"deg = {deg} should be int.")
    if deg <= 0:
        raise ValueError(f"deg = {deg} should be positive.")


def validate_nonnegative(value: float, name: str) -> None:
    """Validate a non-negative scalar generator parameter."""
    if value < 0:
        raise ValueError(f"{name} = {value} should be non-negative.")
