#!/usr/bin/env python
# coding: utf-8
"""
Synthetic data for knapsack problem
"""

import numpy as np


def genData(num_data, num_features, num_items, dim=1, deg=1, noise_width=0, seed=135):
    """
    A function to generate synthetic data and features for knapsack

    Args:
        num_data (int): number of data points
        num_features (int): dimension of features
        num_items (int): number of items
        dim (int): dimension of multi-dimensional knapsack
        deg (int): data polynomial degree
        noise_width (float): half witdth of data random noise
        seed (int): random state seed

    Returns:
       tuple: weights of items (np.ndarray), data features (np.ndarray), costs (np.ndarray)
    """
    # positive integer parameter
    if type(deg) is not int:
        raise ValueError("deg = {} should be int.".format(deg))
    if deg <= 0:
        raise ValueError("deg = {} should be positive.".format(deg))
    # set seed
    rnd = np.random.RandomState(seed)
    # number of data points
    n = num_data
    # dimension of features
    p = num_features
    # dimension of problem
    d = dim
    # number of items
    m = num_items
    # weights of items
    weights = rnd.choice(range(300, 800), size=(d,m)) / 100
    # random matrix parameter B
    B = rnd.binomial(1, 0.5, (m, p))
    # feature vectors
    x = rnd.normal(0, 1, (n, p))
    # value of items
    c = np.zeros((n, m), dtype=int)
    for i in range(n):
        # cost without noise
        values = (np.dot(B, x[i].reshape(p, 1)).T / np.sqrt(p) + 3) ** deg + 1
        # rescale
        values *= 5
        values /= 3.5 ** deg
        # noise
        epislon = rnd.uniform(1 - noise_width, 1 + noise_width, m)
        values *= epislon
        # convert into int
        values = np.ceil(values)
        c[i, :] = values
        # float
        c = c.astype(np.float64)
    return weights, x, c
