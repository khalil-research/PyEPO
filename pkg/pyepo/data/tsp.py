#!/usr/bin/env python
# coding: utf-8
"""
Synthetic data for traveling salesman problem
"""

import numpy as np
from scipy.spatial import distance


def genData(num_data, num_features, num_nodes, deg=1, noise_width=0, seed=135):
    """
    A function to generate synthetic data and features for travelling salesman

    Args:
        num_data (int): number of data points
        num_features (int): dimension of features
        num_nodes (int): number of nodes
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
    # number of data points
    n = num_data
    # dimension of features
    p = num_features
    # number of nodes
    m = num_nodes
    # random coordinates
    coords = np.concatenate((rnd.uniform(-2, 2, (m // 2, 2)),
                             rnd.normal(0, 1, (m - m // 2, 2))))
    # distance matrix
    org_dist = distance.cdist(coords, coords, "euclidean")
    # random matrix parameter B
    B = rnd.binomial(1, 0.5, (m * (m - 1) // 2, p)) * rnd.uniform(
        -2, 2, (m * (m - 1) // 2, p))
    # feature vectors
    x = rnd.normal(0, 1, (n, p))
    # init cost
    c = np.zeros((n, m * (m - 1) // 2))
    for i in range(n):
        # reshape
        l = 0
        for j in range(m):
            for k in range(j + 1, m):
                c[i, l] = org_dist[j, k]
                l += 1
        # noise
        noise = rnd.uniform(1 - noise_width, 1 + noise_width,
                                  m * (m - 1) // 2)
        # from feature to edge
        c[i] += (((np.dot(B, x[i].reshape(p, 1)).T / np.sqrt(p) + 3)
                  ** deg) / 3 ** (deg - 1)).reshape(-1) * noise
    # rounding
    c = np.around(c, decimals=4)
    return x, c
