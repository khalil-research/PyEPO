"""
pyepo
=====
PyTorch-based End-to-End Predict-then-Optimize Tool
"""

import logging
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyepo")
except PackageNotFoundError:
    # package not installed (e.g. running from source tree without install)
    __version__ = "0.0.0+unknown"

# Silence library logs by default; users opt in with logging.basicConfig(level=INFO).
logging.getLogger(__name__).addHandler(logging.NullHandler())

from pyepo import EPO, data, func, metric, model, twostage
from pyepo.EPO import MAXIMIZE, MINIMIZE, ModelSense

__all__ = [
    "EPO",
    "MAXIMIZE",
    "MINIMIZE",
    "ModelSense",
    "__version__",
    "data",
    "func",
    "metric",
    "model",
    "twostage",
]
