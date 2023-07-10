#!/usr/bin/env python
# coding: utf-8
"""
Metrics for SKlearn model
"""

import numpy as np

from pyepo.utlis import getArgs
from pyepo import EPO

def SPOError(pred_cost, true_cost, model_type, args):
    """
    A function to calculate normalized true regret

    Args:
        pred_cost (numpy.array): predicted costs
        true_cost (numpy.array): true costs
        model_type (ABCMeta): optModel class type
        args (dict): optModel args

    Returns:
        float: regret loss
    """
    pred_cost = np.array(pred_cost)
    true_cost = np.array(true_cost)
    assert pred_cost.shape == true_cost.shape, \
    "Shape of true and predicted value does not match."
    # rebuild model
    optmodel = model_type(**args)
    # init sum
    regret_sum = 0
    optobj_sum = 0
    for c, cp in zip(true_cost, pred_cost):
        # opt sol for pred cost
        optmodel.setObj(cp)
        sol, _ = optmodel.solve()
        # obj with true cost
        obj = np.dot(sol, c)
        # opt obj for true cost
        optmodel.setObj(c)
        _, optobj = optmodel.solve()
        # calculate regret
        if optmodel.modelSense == EPO.MINIMIZE:
            regret = obj - optobj
        if optmodel.modelSense == EPO.MAXIMIZE:
            regret = optobj - obj
        regret_sum += regret
        optobj_sum += np.abs(optobj)
    # normalized regret
    norm_regret = regret_sum / (optobj_sum + 1e-7)
    return norm_regret


def makeSkScorer(optmodel):
    """
    A function to create sklearn scorer

    Args:
        optmodel (optModel): optimization model

    Returns:
        scorer: callable object that returns a scalar score; less is better.
    """
    from sklearn.metrics import make_scorer
    # get class
    model_type = type(optmodel)
    # get args
    args = getArgs(optmodel)
    # build score
    SPO_scorer = make_scorer(SPOError, greater_is_better=False,
                             model_type=model_type, args=args)
    return SPO_scorer


def makeAutoSkScorer(optmodel):
    """
    A function to create Auto-SKlearn scorer

    Args:
        optmodel (optModel): optimization model

    Returns:
        scorer: callable object that returns a scalar score; less is better.
    """
    from autosklearn.metrics import make_scorer
    # get class
    model_type = type(optmodel)
    # get args
    args = getArgs(optmodel)
    # build score
    SPO_scorer = make_scorer(name='SPO_error',
                             score_func=SPOError,
                             greater_is_better=False,
                             needs_proba=False,
                             needs_threshold=False,
                             model_type=model_type,
                             args=args)
    return SPO_scorer


def testMSE(pred_cost, true_cost, model_type, args):
    """
    A function to calculate MSE for testing

    Args:
        pred_cost (array): predicted costs
        true_cost (array): true costs
        model_type (ABCMeta): optModel class type
        args (dict): optModel args

    Returns:
        float: mse
    """
    pred_cost = np.array(pred_cost)
    true_cost = np.array(true_cost)
    assert pred_cost.shape == true_cost.shape, \
    "Shape of true and predicted value does not match."
    # rebuild model
    optmodel = model_type(**args)
    # init sum
    regret_sum = 0
    optobj_sum = 0
    for i in range(pred_cost.shape[0]):
        # opt sol for pred cost
        print(pred_cost.shape[0])
    return np.square(pred_cost - true_cost).mean()


def makeTestMSEScorer(optmodel):
    """
    A function to create MSE scorer for testing

    Args:
        optmodel (optModel): optimization model

    Returns:
        scorer: callable object that returns a scalar score; less is better.
    """
    from autosklearn.metrics import make_scorer
    # get class
    model_type = type(optmodel)
    # get args
    args = getArgs(optmodel)
    # build score
    mse_scorer = make_scorer(name='test_error',
                             score_func=testMSE,
                             greater_is_better=False,
                             model_type=model_type,
                             args=args)
    return mse_scorer
