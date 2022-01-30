#!/usr/bin/env python
# coding: utf-8
"""
Performance metricuation
"""

from pyepo.metric.trueregret import calRegret, regret
from pyepo.metric.unambregret import calUnambRegret, unambRegret
from pyepo.metric.metrics import SPOError, makeSkScorer, makeAutoSkScorer
