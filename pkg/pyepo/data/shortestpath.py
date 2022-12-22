#!/usr/bin/env python
# coding: utf-8
"""
Synthetic data for Shortest path problem
"""

import numpy as np


def genData(num_data, num_features, grid, deg=1, noise_width=0, seed=135):
    """
    A function to generate synthetic data and features for shortest path

    Args:
        num_data (int): number of data points
        num_features (int): dimension of features
        grid (int, int): size of grid network
        deg (int): data polynomial degree
        noise_width (float): half witdth of data random noise
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
    # numbrnda points
    n = num_data
    # dimension of features
    p = num_features
    # dimension of the cost vector
    d = (grid[0] - 1) * grid[1] + (grid[1] - 1) * grid[0]
    # random matrix parameter B
    B = rnd.binomial(1, 0.5, (d, p))
    # feature vectors
    x = rnd.normal(0, 1, (n, p))
    # cost vectors
    c = np.zeros((n, d))
    for i in range(n):
        # cost without noise
        ci = (np.dot(B, x[i].reshape(p, 1)).T / np.sqrt(p) + 3) ** deg + 1
        # rescale
        ci /= 3.5 ** deg
        # noise
        epislon = rnd.uniform(1 - noise_width, 1 + noise_width, d)
        ci *= epislon
        c[i, :] = ci

    return x, c
