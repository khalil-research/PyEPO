#!/usr/bin/env python
"""
Performance evaluation
"""

from pyepo.metric.metrics import SPOError, makeAutoSkScorer, makeSkScorer
from pyepo.metric.mse import MSE
from pyepo.metric.regret import calRegret, regret
from pyepo.metric.unambregret import calUnambRegret, unambRegret
