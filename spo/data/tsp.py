#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.spatial import distance

def genData(num_data, num_features, num_nodes, deg=1, noise_width=0, seed=135):
    """
    generate synthetic data and features for traveling salesman
    Args:
        num_data: number of data points
        num_features: dimension of features
        num_nodes: number of nodes
        deg: a fixed positive integer parameter
        noise_withd: half witdth of random noise
        seed: random seed
    """
    # positive integer parameter
    assert type(deg) is int, 'deg = {} should be int.'.format(deg)
    assert deg > 0, 'deg = {} should be positive.'.format(deg)
    # set seed
    np.random.seed(seed)
    # number of data points
    n = num_data
    # dimension of features
    p = num_features
    # number of nodes
    m = num_nodes
    # random coordinates
    coords = np.concatenate((np.random.uniform(-2, 2, (m//2,2)),
                             np.random.normal(0, 1, (m-m//2,2))))
    # distance matrix
    org_dist = distance.cdist(coords, coords, 'euclidean') * 3 ** deg
    # random matrix parameter B
    B = np.random.binomial(1, 0.5, (m*(m-1)//2,p)) * np.random.uniform(-2, 2, (m*(m-1)//2,p))
    # feature vectors
    x = np.random.normal(0, 1, (n,p))
    # init cost
    c = np.zeros((n, m*(m-1)//2))
    for i in range(n):
        # reshape
        l = 0
        for j in range(m):
            for k in range(j+1, m):
                c[i,l] = org_dist[j,k]
                l += 1
        # from feature to edge
        c[i] += ((np.dot(B, x[i].reshape(p,1)).T / np.sqrt(p) + 3) ** deg).reshape(-1)
        # rescale
        c[i] /= 3 ** (deg - 1)
        # noise
        noise =  np.random.uniform(1-noise_width, 1+noise_width, m*(m-1)//2)
        c[i] = c[i] * noise
    # rounding
    c = np.around(c, decimals=4)
    return x, c
