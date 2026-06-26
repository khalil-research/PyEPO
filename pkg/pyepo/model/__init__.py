"""
Optimization Model based on solvers
"""

from importlib import import_module

from pyepo.model import opt
from pyepo.model.opt import ModelSpec
from pyepo.model.predefined import (
    knapsackModel,
    portfolioModel,
    shortestPathModel,
    tspModel,
    vrpModel,
)

__all__ = [
    "ModelSpec",
    "knapsackModel",
    "opt",
    "portfolioModel",
    "shortestPathModel",
    "tspModel",
    "vrpModel",
]


def _try_import_backend(name: str) -> bool:
    """Expose an optional solver backend if its dependencies are available."""
    try:
        globals()[name] = import_module(f"pyepo.model.{name}")
    except ImportError:
        return False
    return True


for _backend in ("grb", "copt", "omo", "mpax", "ort"):
    if _try_import_backend(_backend):
        __all__ += [_backend]

del _backend, _try_import_backend
