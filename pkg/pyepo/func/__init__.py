"""
Pytorch autograd function for end-to-end training
"""

from pyepo.func.blackbox import blackboxOpt, negativeIdentity
from pyepo.func.contrastive import NCE, contrastiveMAP
from pyepo.func.perturbed import (
    adaptiveImplicitMLE,
    implicitMLE,
    perturbedFenchelYoung,
    perturbedFenchelYoungMul,
    perturbedOpt,
    perturbedOptMul,
)
from pyepo.func.rank import listwiseLTR, pairwiseLTR, pointwiseLTR
from pyepo.func.surrogate import SPOPlus, perturbationGradient

__all__ = [
    "NCE",
    "SPOPlus",
    "adaptiveImplicitMLE",
    "blackboxOpt",
    "contrastiveMAP",
    "implicitMLE",
    "listwiseLTR",
    "negativeIdentity",
    "pairwiseLTR",
    "perturbationGradient",
    "perturbedFenchelYoung",
    "perturbedFenchelYoungMul",
    "perturbedOpt",
    "perturbedOptMul",
    "pointwiseLTR",
]
