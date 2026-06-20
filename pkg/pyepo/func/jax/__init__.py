"""
JAX autograd function for end-to-end training
"""

import importlib

try:
    import jax  # noqa: F401

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

# public name -> submodule that defines it (each pair is full name, acronym)
_EXPORTS = {
    "smartPredictThenOptimizePlus": "surrogate",
    "SPOPlus": "surrogate",
    "perturbationGradient": "surrogate",
    "PG": "surrogate",
    "blackboxOpt": "blackbox",
    "DBB": "blackbox",
    "negativeIdentity": "blackbox",
    "NID": "blackbox",
    "perturbedOpt": "perturbed",
    "DPO": "perturbed",
    "perturbedOptMul": "perturbed",
    "DPOMul": "perturbed",
    "perturbedFenchelYoung": "perturbed",
    "PFY": "perturbed",
    "perturbedFenchelYoungMul": "perturbed",
    "PFYMul": "perturbed",
    "implicitMLE": "perturbed",
    "IMLE": "perturbed",
    "adaptiveImplicitMLE": "perturbed",
    "AIMLE": "perturbed",
    "regularizedFrankWolfeOpt": "regularized",
    "RFWO": "regularized",
    "regularizedFrankWolfeFenchelYoung": "regularized",
    "RFY": "regularized",
    "noiseContrastiveEstimation": "contrastive",
    "NCE": "contrastive",
    "contrastiveMAP": "contrastive",
    "CMAP": "contrastive",
    "coneAlignedCosine": "cave",
    "CaVE": "cave",
    "listwiseLearningToRank": "rank",
    "lsLTR": "rank",
    "pairwiseLearningToRank": "rank",
    "prLTR": "rank",
    "pointwiseLearningToRank": "rank",
    "ptLTR": "rank",
}

__all__ = list(_EXPORTS)  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name):
    if not _HAS_JAX:
        raise ImportError(
            "pyepo.func.jax requires JAX. Install with `pip install pyepo[mpax]` "
            "(MPAX), or any JAX install for the pure_callback path."
        )
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(name)
    return getattr(importlib.import_module(f"pyepo.func.jax.{mod}"), name)
