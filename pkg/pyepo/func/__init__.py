#!/usr/bin/env python
# coding: utf-8
"""
Pytorch autograd function for SPO training
"""

from pyepo.func.blackbox import blackboxOpt
from pyepo.func.spoplus import SPOPlus
from pyepo.func.perturbed import perturbedOpt, perturbedFenchelYoung
from pyepo.func.contrastive import NCE, contrastiveMAP
from pyepo.func.rank import listwiseLTR, pairwiseLTR, pointwiseLTR
