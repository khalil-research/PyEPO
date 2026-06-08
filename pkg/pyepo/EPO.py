"""
Constants
"""

from __future__ import annotations

from enum import Enum, IntEnum


class ModelSense(IntEnum):
    """Optimization direction for the objective."""

    MINIMIZE = 1
    MAXIMIZE = -1


class VarType(Enum):
    """Decision-variable type."""

    BINARY = "B"
    INTEGER = "I"
    CONTINUOUS = "C"


# IntEnum is int, so `EPO.MINIMIZE == 1` holds
MINIMIZE = ModelSense.MINIMIZE
MAXIMIZE = ModelSense.MAXIMIZE
BINARY = VarType.BINARY
INTEGER = VarType.INTEGER
CONTINUOUS = VarType.CONTINUOUS


__all__ = [
    "BINARY",
    "CONTINUOUS",
    "INTEGER",
    "MAXIMIZE",
    "MINIMIZE",
    "ModelSense",
    "VarType",
]
