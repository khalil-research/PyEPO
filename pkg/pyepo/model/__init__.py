"""
Optimization Model based on solvers
"""

from pyepo.model import opt
from pyepo.model.predefined import (
    knapsackModel,
    portfolioModel,
    shortestPathModel,
    tspModel,
    vrpModel,
)

__all__ = [
    "knapsackModel",
    "opt",
    "portfolioModel",
    "shortestPathModel",
    "tspModel",
    "vrpModel",
]

try:
    from pyepo.model import grb

    __all__ += ["grb"]
except ImportError:
    pass
try:
    from pyepo.model import copt

    __all__ += ["copt"]
except ImportError:
    pass
try:
    from pyepo.model import omo

    __all__ += ["omo"]
except ImportError:
    pass
try:
    from pyepo.model import mpax

    __all__ += ["mpax"]
except ImportError:
    pass
try:
    from pyepo.model import ort

    __all__ += ["ort"]
except ImportError:
    pass
