"""
Constants
"""

from __future__ import annotations

from enum import IntEnum


class ModelSense(IntEnum):
    """Optimization direction for the objective."""

    MINIMIZE = 1
    MAXIMIZE = -1


# IntEnum is int, so `EPO.MINIMIZE == 1` holds
MINIMIZE = ModelSense.MINIMIZE
MAXIMIZE = ModelSense.MAXIMIZE


__all__ = ["MAXIMIZE", "MINIMIZE", "ModelSense"]
