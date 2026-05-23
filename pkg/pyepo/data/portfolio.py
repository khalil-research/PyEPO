#!/usr/bin/env python
"""
Synthetic data for portfolio
"""

from __future__ import annotations

import numpy as np


def genData(
    num_data: int,
    num_features: int,
    num_assets: int,
    deg: int = 1,
    noise_level: float = 1,
    seed: int = 135,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A function to generate synthetic data and features for portfolio

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
    if not isinstance(deg, int):
        raise ValueError(f"deg = {deg} should be int.")
    if deg <= 0:
        raise ValueError(f"deg = {deg} should be positive.")
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
    L = rnd.uniform(-2.5e-3 * noise_level, 2.5e-3 * noise_level, (num_assets, num_features))
    # feature vectors
    x = rnd.normal(0, 1, (n, p))
    # mean return of assets
    r = np.zeros((n, m))
    for i in range(n):
        # mean return of assets
        r[i] = (0.05 * np.dot(B, x[i].reshape(p, 1)).T / np.sqrt(p) + 0.1 ** (1 / deg)) ** deg
        # random noise
        f = rnd.randn(num_features)
        eps = rnd.randn(num_assets)
        r[i] += L @ f + 0.01 * noise_level * eps
    # covariance matrix of the returns
    cov = L @ L.T + (1e-2 * noise_level) ** 2 * np.eye(num_assets)
    return cov, x, r
