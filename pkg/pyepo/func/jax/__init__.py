"""
JAX autograd function for end-to-end training
"""

try:
    import jax  # noqa: F401

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

__all__ = ["SPOPlus", "smartPredictThenOptimizePlus"]


def __getattr__(name):
    if not _HAS_JAX:
        raise ImportError(
            "pyepo.func.jax requires JAX. Install with `pip install pyepo[mpax]` "
            "(MPAX), or any JAX install for the pure_callback path."
        )
    if name in ("SPOPlus", "smartPredictThenOptimizePlus"):
        from pyepo.func.jax.surrogate import SPOPlus

        return SPOPlus
    raise AttributeError(name)
