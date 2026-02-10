#!/usr/bin/env python
# coding: utf-8
"""
Synthetic data for traveling salesman problem
"""

import numpy as np
from scipy.spatial import distance


def genData(num_data, num_features, num_nodes, deg=1, noise_width=0, seed=135):
    """
    A function to generate synthetic data and features for traveling salesman

    Args:
        num_data (int): number of data points
        num_features (int): dimension of features
        num_nodes (int): number of nodes
        deg (int): data polynomial degree
        noise_width (float): half width of data random noise
        seed (int): random seed

    Returns:
        tuple: data features (np.ndarray), costs (np.ndarray)
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
    # extract upper triangle distances (vectorized)
    triu_idx = np.triu_indices(m, k=1)
    dist_upper = org_dist[triu_idx]
    # base distance for all data points
    c = np.tile(dist_upper, (n, 1))
    # feature-based cost
    feature_cost = ((x @ B.T / np.sqrt(p) + 3) ** deg) / 3 ** (deg - 1)
    # noise
    noise = rnd.uniform(1 - noise_width, 1 + noise_width, (n, m * (m - 1) // 2))
    c += feature_cost * noise
    # rounding
    c = np.around(c, decimals=4)
    return x, c
