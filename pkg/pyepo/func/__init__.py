#!/usr/bin/env python
# coding: utf-8
"""
Pytorch autograd function for end-to-end training
"""

from pyepo.func.spoplus import SPOPlus
from pyepo.func.blackbox import blackboxOpt, negativeIdentity
from pyepo.func.perturbed import perturbedOpt, perturbedFenchelYoung, implicitMLE
from pyepo.func.contrastive import NCE, contrastiveMAP
from pyepo.func.rank import listwiseLTR, pairwiseLTR, pointwiseLTR
