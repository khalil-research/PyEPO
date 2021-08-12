#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.base import clone

class sklearnPred:
    """
    Two-stage prediction and optimization with scikit-learn.

    Args:
        pmodel (sklearn.base.BaseEstimator): scikit-learn regression model
        omodel (optModel): optimization model
    """

    def __init__(self, pmodel, omodel):
        self.predictor = [clone(pmodel) for _ in range(omodel.num_cost)]
        self.optimizer = omodel
        self.trained = False

    def fit(self, x, c):
        """
        A method to fit training data

        Args:
            x (ndarray): features
            c (ndarray): costs for objective fucntion
        """
        assert len(c.shape) == 2 and c.shape[-1] == self.optimizer.num_cost, 'Dimension of cost does not macth.'
        for j in range(self.optimizer.num_cost):
            self.predictor[j].fit(x, c[:,j])
        self.trained = True

    def predict(self, x):
        """
        A method to predict

        Args:
            x (ndarray): features

        Returns:
            ndarray: predicted cost
        """
        cp = np.zeros((x.shape[0],0))
        assert self.trained, 'This two-stage sklearnPred instance is not fitted yet.'
        for j in range(self.optimizer.num_cost):
            cp = np.concatenate((cp, self.predictor[j].predict(x).reshape(-1,1)), axis=1)
        return cp
