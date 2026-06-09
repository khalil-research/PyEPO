"""
Pytorch autograd function for end-to-end training
"""

from pyepo.func.blackbox import DBB, NID, blackboxOpt, negativeIdentity
from pyepo.func.cave import CaVE, coneAlignedCosine
from pyepo.func.contrastive import CMAP, NCE, contrastiveMAP, noiseContrastiveEstimation
from pyepo.func.perturbed import (
    AIMLE,
    DPO,
    IMLE,
    PFY,
    DPOMul,
    PFYMul,
    adaptiveImplicitMLE,
    implicitMLE,
    perturbedFenchelYoung,
    perturbedFenchelYoungMul,
    perturbedOpt,
    perturbedOptMul,
)
from pyepo.func.rank import (
    listwiseLearningToRank,
    lsLTR,
    pairwiseLearningToRank,
    pointwiseLearningToRank,
    prLTR,
    ptLTR,
)
from pyepo.func.regularized import (
    RFWO,
    RFY,
    regularizedFrankWolfeFenchelYoung,
    regularizedFrankWolfeOpt,
)
from pyepo.func.surrogate import PG, SPOPlus, perturbationGradient, smartPredictThenOptimizePlus

# each line is full name, acronym
__all__ = [
    "smartPredictThenOptimizePlus",
    "SPOPlus",
    "perturbationGradient",
    "PG",
    "blackboxOpt",
    "DBB",
    "negativeIdentity",
    "NID",
    "perturbedOpt",
    "DPO",
    "perturbedOptMul",
    "DPOMul",
    "perturbedFenchelYoung",
    "PFY",
    "perturbedFenchelYoungMul",
    "PFYMul",
    "implicitMLE",
    "IMLE",
    "adaptiveImplicitMLE",
    "AIMLE",
    "regularizedFrankWolfeOpt",
    "RFWO",
    "regularizedFrankWolfeFenchelYoung",
    "RFY",
    "noiseContrastiveEstimation",
    "NCE",
    "contrastiveMAP",
    "CMAP",
    "coneAlignedCosine",
    "CaVE",
    "listwiseLearningToRank",
    "lsLTR",
    "pairwiseLearningToRank",
    "prLTR",
    "pointwiseLearningToRank",
    "ptLTR",
]
