#!/usr/bin/env python
# coding: utf-8

import numpy as np

def genData(num_data, num_features, num_surgeries, deg=1, noise_width=0, seed=135):
    """
    A function to generate synthetic data and features for surgery scheduling

    Args:
        num_data (int): number of data points
        num_features (int): dimension of features
        num_surgeries (int): number of surgeries
        deg (int): data polynomial degree
        noise_withd (float): half witdth of data random noise
        seed (int): random seed

    Returns:
        tuple: data features (ndarray), costs (ndarray)
    """
    # set seed
    np.random.seed(seed)
    # number of data points
    n = num_data
    # dimension of features
    p = num_features
    # dimension of the cost vector
    d = num_surgeries
    # random matrix parameter B*
    B = np.random.binomial(1, 0.5, (d,p))
    # positive integer parameter
    assert type(deg) is int, 'deg = {} should be int.'.format(deg)
    assert deg > 0, 'deg = {} should be positive.'.format(deg)
    # feature vectors
    x = np.random.normal(0, 1, (n,p))
    # cost vectors
    c = np.zeros((n,d))
    for i in range(n):
        # cost without noise
        ci = (np.dot(B, x[i].reshape(p,1)).T / np.sqrt(p) + 3) ** deg + 1
        # noise
        epislon = np.random.uniform(1-noise_width, 1+noise_width, d)
        ci *= epislon
        c[i,:] = ci

    return x, c
