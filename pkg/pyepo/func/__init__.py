"""
Pytorch autograd function for end-to-end training
"""

from pyepo.func.blackbox import blackboxOpt, negativeIdentity
from pyepo.func.cave import coneAlignedCosine
from pyepo.func.contrastive import contrastiveMAP, noiseContrastiveEstimation
from pyepo.func.perturbed import (
    adaptiveImplicitMLE,
    implicitMLE,
    perturbedFenchelYoung,
    perturbedFenchelYoungMul,
    perturbedOpt,
    perturbedOptMul,
)
from pyepo.func.rank import listwiseLTR, pairwiseLTR, pointwiseLTR
from pyepo.func.regularized import regularizedFrankWolfeFenchelYoung, regularizedFrankWolfeOpt
from pyepo.func.surrogate import SPOPlus, perturbationGradient

__all__ = [
    "SPOPlus",
    "blackboxOpt",
    "negativeIdentity",
    "perturbedOpt",
    "perturbedOptMul",
    "perturbedFenchelYoung",
    "perturbedFenchelYoungMul",
    "regularizedFrankWolfeOpt",
    "regularizedFrankWolfeFenchelYoung",
    "noiseContrastiveEstimation",
    "contrastiveMAP",
    "pointwiseLTR",
    "pairwiseLTR",
    "listwiseLTR",
    "implicitMLE",
    "adaptiveImplicitMLE",
    "perturbationGradient",
    "coneAlignedCosine",
]
