#!/usr/bin/env python
# coding: utf-8

import numpy as np

def genData(num_data, num_features, num_items, deg=1, noise_width=0, cor=False, seed=135):
    """
    generate synthetic data and features for shortest path
    Args:
        num_data: number of data points
        num_features: dimension of features
        num_items: number of items
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
    # number of items
    m = num_items
    # weights of items
    weights = np.random.choice(range(3,8), size=m)
    # random matrix parameter B
    B = np.random.binomial(1, 0.5, (m,p))
    # feature vectors
    x = np.random.normal(0, 1, (n,p))
    # value of items
    c = np.zeros((n,m), dtype=int)
    for i in range(n):
        # cost without noise
        values = ((np.dot(B, x[i].reshape(p,1)).T / np.sqrt(p) + 3) ** deg + 1) / 3 ** deg * 5
        # correlation with weights
        if cor:
            values += weights - 3
        # noise
        epislon = np.random.uniform(1-noise_width, 1+noise_width, m)
        values *= epislon
        # convert into int
        values = np.ceil(values).astype(int)
        c[i,:] = values
    return weights, x, c
