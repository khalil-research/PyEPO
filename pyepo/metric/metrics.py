#!/usr/bin/env python
# coding: utf-8
"""
Metrics for SKlearn model
"""

import numpy as np

from pyepo.utlis import getArgs

def SPOError(pred_cost, true_cost, model_type, args):
    """
    A function to calculate normalized true regret

    Args:
        pred_cost (array): predicted costs
        true_cost (array): true costs
        model_type (ABCMeta): optModel class type
        args (dict): optModel args

    Returns:
        float: true SPO losses
    """
    pred_cost = np.array(pred_cost)
    true_cost = np.array(true_cost)
    assert pred_cost.shape == true_cost.shape, \
    "Shape of true and predicted value does not match."
    # rebuild model
    omodel = model_type(**args)
    # init sum
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


def makeSkScorer(omodel):
    """
    A function to create sklearn scorer

    Args:
        omodel (optModel): optimization model

    Returns:
        scorer: callable object that returns a scalar score; less is better.
    """
    from sklearn.metrics import make_scorer
    # get class
    model_type = type(omodel)
    # get args
    args = getArgs(omodel)
    # build score
    SPO_scorer = make_scorer(SPOError, greater_is_better=False,
                             model_type=model_type, args=args)
    return SPO_scorer


def makeAutoSkScorer(omodel):
    """
    A function to create Auto-SKlearn scorer

    Args:
        omodel (optModel): optimization model

    Returns:
        scorer: callable object that returns a scalar score; less is better.
    """
    from autosklearn.metrics import make_scorer
    # get class
    model_type = type(omodel)
    # get args
    args = getArgs(omodel)
    # build score
    SPO_scorer = make_scorer(name='SPO_error',
                             score_func=SPOError,
                             greater_is_better=False,
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
    omodel = model_type(**args)
    # init sum
    regret_sum = 0
    optobj_sum = 0
    for i in range(pred_cost.shape[0]):
        # opt sol for pred cost
        print(pred_cost.shape[0])
    return np.square(pred_cost - true_cost).mean()


def makeTestMSEScorer(omodel):
    """
    A function to create MSE scorer for testing

    Args:
        omodel (optModel): optimization model

    Returns:
        scorer: callable object that returns a scalar score; less is better.
    """
    from autosklearn.metrics import make_scorer
    # get class
    model_type = type(omodel)
    # get args
    args = getArgs(omodel)
    # build score
    mse_scorer = make_scorer(name='test_error',
                             score_func=testMSE,
                             greater_is_better=False,
                             model_type=model_type,
                             args=args)
    return mse_scorer
