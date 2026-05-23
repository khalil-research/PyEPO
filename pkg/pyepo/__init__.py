#!/usr/bin/env python

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

import pyepo.data
import pyepo.func
import pyepo.metric
import pyepo.model
import pyepo.twostage
