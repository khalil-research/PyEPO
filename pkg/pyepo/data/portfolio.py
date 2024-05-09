#!/usr/bin/env python
# coding: utf-8
"""
Synthetic data for portfolio
"""

import numpy as np


def genData(num_data, num_features, num_assets, deg=1, noise_level=1, seed=135):
    """
    A function to generate synthetic data and features for travelling salesman

    Args:
        num_data (int): number of data points
        num_features (int): dimension of features
        num_assets (int): number of assets
        deg (int): data polynomial degree
        noise_level (float): level of data random noise
        seed (int): random seed

    Returns:
        tuple: data features (np.ndarray), costs (np.ndarray)
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
    # number of assets
    m = num_assets
    # random matrix parameter B
    B = rnd.binomial(1, 0.5, (m, p))
    # random matrix parameter L
    L = rnd.uniform(-2.5e-3*noise_level, 2.5e-3*noise_level, (num_assets, num_features))
    # feature vectors
    x = rnd.normal(0, 1, (n, p))
    # value of items
    r = np.zeros((n, m))
    for i in range(n):
        # mean return of assets
        r[i] = (0.05 * np.dot(B, x[i].reshape(p, 1)).T / np.sqrt(p) + \
                0.1 ** (1 / deg)) ** deg
        # random noise
        f = rnd.randn(num_features)
        eps = rnd.randn(num_assets)
        r[i] += L @ f + 0.01 * noise_level * eps
    # covariance matrix of the returns
    cov = L @ L.T + (1e-2 * noise_level) ** 2 * np.eye(num_assets)
    return cov, x, r
