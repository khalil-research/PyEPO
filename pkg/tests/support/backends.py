"""Optional-backend probes and reusable pytest skip markers."""

from __future__ import annotations

import pytest
import torch

try:
    from pyepo.model.grb.shortestpath import shortestPathModel

    shortestPathModel(grid=(3, 3))
    _HAS_GUROBI = True
except Exception:  # noqa: BLE001
    _HAS_GUROBI = False

try:
    import pyomo.environ  # noqa: F401

    _HAS_PYOMO = True
except Exception:  # noqa: BLE001
    _HAS_PYOMO = False

try:
    import coptpy  # noqa: F401

    _HAS_COPT = True
except Exception:  # noqa: BLE001
    _HAS_COPT = False

try:
    from pyepo.model.ort.ortmodel import _HAS_ORTOOLS
except Exception:  # noqa: BLE001
    _HAS_ORTOOLS = False

try:
    import jax  # noqa: F401

    _HAS_JAX = True
except Exception:  # noqa: BLE001
    _HAS_JAX = False

try:
    import mpax  # noqa: F401

    _HAS_MPAX = _HAS_JAX
except Exception:  # noqa: BLE001
    _HAS_MPAX = False

try:
    import flax  # noqa: F401

    _HAS_FLAX = _HAS_JAX
except Exception:  # noqa: BLE001
    _HAS_FLAX = False

try:
    from pyepo.twostage.autosklearnpred import _HAS_AUTO
except Exception:  # noqa: BLE001
    _HAS_AUTO = False

try:
    import clarabel  # noqa: F401

    _HAS_CLARABEL = True
except Exception:  # noqa: BLE001
    _HAS_CLARABEL = False

_HAS_CUDA = torch.cuda.is_available()

try:
    import jax as _jax

    _HAS_JAX_GPU = any(d.platform == "gpu" for d in _jax.devices())
except Exception:  # noqa: BLE001
    _HAS_JAX_GPU = False

requires_gurobi = pytest.mark.skipif(not _HAS_GUROBI, reason="Gurobi not installed")
requires_pyomo = pytest.mark.skipif(not _HAS_PYOMO, reason="Pyomo not installed")
requires_copt = pytest.mark.skipif(not _HAS_COPT, reason="COPT not installed")
requires_ortools = pytest.mark.skipif(not _HAS_ORTOOLS, reason="OR-Tools not installed")
requires_jax = pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
requires_mpax = pytest.mark.skipif(not _HAS_MPAX, reason="MPAX (jax + mpax) not installed")
requires_flax = pytest.mark.skipif(not _HAS_FLAX, reason="Flax (jax + flax) not installed")
requires_clarabel = pytest.mark.skipif(not _HAS_CLARABEL, reason="Clarabel not installed")
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")
requires_jax_gpu = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_JAX_GPU), reason="CUDA + jax-gpu not both available"
)

__all__ = [
    "_HAS_AUTO",
    "_HAS_CLARABEL",
    "_HAS_COPT",
    "_HAS_CUDA",
    "_HAS_FLAX",
    "_HAS_GUROBI",
    "_HAS_JAX",
    "_HAS_JAX_GPU",
    "_HAS_MPAX",
    "_HAS_ORTOOLS",
    "_HAS_PYOMO",
    "requires_clarabel",
    "requires_copt",
    "requires_cuda",
    "requires_flax",
    "requires_gurobi",
    "requires_jax",
    "requires_jax_gpu",
    "requires_mpax",
    "requires_ortools",
    "requires_pyomo",
]
