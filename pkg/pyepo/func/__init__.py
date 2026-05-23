#!/usr/bin/env python
"""
Pytorch autograd function for end-to-end training
"""

from pyepo.func.blackbox import blackboxOpt, negativeIdentity
from pyepo.func.contrastive import NCE, contrastiveMAP
from pyepo.func.perturbed import (
    adaptiveImplicitMLE,
    implicitMLE,
    perturbedFenchelYoung,
    perturbedOpt,
)
from pyepo.func.rank import listwiseLTR, pairwiseLTR, pointwiseLTR
from pyepo.func.surrogate import SPOPlus, perturbationGradient
