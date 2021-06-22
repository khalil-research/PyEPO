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
        seed: random seeds
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
    coords = np.random.uniform(0, 10, (m,2))
    # distance matrix
    org_dist = distance.cdist(coords, coords, 'euclidean')
    # random matrix parameter B
    B = np.random.binomial(1, 0.5, (m,p)) * np.random.uniform(0, 1, (m,p))
    # feature vectors
    x = np.random.normal(0, 1, (n,p))
    # init cost
    c = np.repeat(org_dist.reshape(1,m,m), n, axis=0)
    for i in range(n):
        adds = ((np.dot(B, x[i].reshape(p,1)).T / np.sqrt(p) + 3) ** deg + 1).reshape(-1) / 3 ** deg * 3
        for j in range(m):
            for k in range(m):
                if j == k:
                    continue
                c[i,j,k] += adds[j] + adds[k]
        # noise
        a =  np.random.uniform(1-noise_width, 1+noise_width, (m,m))
        noise = np.tril(a) + np.tril(a, -1).T
        c[i] = c[i] * noise
    return x, c
