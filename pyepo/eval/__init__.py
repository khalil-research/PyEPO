#!/usr/bin/env python
# coding: utf-8
"""
Performance evaluation
"""

from pyepo.eval.trueregret import calRegret, regret
from pyepo.eval.unambregret import calUnambRegret, unambRegret
from pyepo.eval.metrics import SPOError, makeSkScorer, makeAutoSkScorer
