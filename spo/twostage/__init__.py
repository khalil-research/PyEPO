#!/usr/bin/env python
# coding: utf-8
"""
Two-stage predict then optimize model
"""

from spo.twostage.sklearnpred import sklearnPred
try:
    from spo.twostage.autosklearnpred import autoSklearnPred
except:
    print("Auto-Sklearn is not included.")
