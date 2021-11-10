#!/usr/bin/env python
# coding: utf-8
"""
metrics for SKlearn model
"""

import numpy as np
import sklearn
#import autosklearn.metrics

def SPOError(pred_cost, true_cost, omodel):
    """
    A function to calculate normalized true SPO
    """
    assert pred_cost.shape == true_cost.shape, \
    "Shape of true and predicted value does not match."
    regret_sum = 0
    optobj_sum = 0
    for i in range(pred_cost.shape[0]):
        # opt sol for pred cost
        omodel.setObj(pred_cost[i])
        sol, _ = omodel.solve()
        # obj with true cost
        obj = np.dot(sol, true_cost[i])
        # opt obj for true cost
        omodel.setObj(true_cost[i])
        _, optobj = omodel.solve()
        # calculate regret
        regret = obj - optobj
        regret_sum += regret
        optobj_sum += np.abs(optobj)
    return regret_sum / (optobj_sum + 1e-7)
