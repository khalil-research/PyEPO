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
        noise_width (float): half width of data random noise
        seed (int): random state seed

    Returns:
       tuple: weights of items (np.ndarray), data features (np.ndarray), costs (np.ndarray)
    """
    # positive integer parameter
    if not isinstance(deg, int):
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
    # value of items (vectorized)
    c = (x @ B.T / np.sqrt(p) + 3) ** deg + 1
    # rescale
    c *= 5
    c /= 3.5 ** deg
    # noise
    epsilon = rnd.uniform(1 - noise_width, 1 + noise_width, (n, m))
    c *= epsilon
    # convert into int
    c = np.ceil(c).astype(np.float64)
    return weights, x, c
