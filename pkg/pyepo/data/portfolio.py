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
        num_data: number of data points
        num_features: dimension of features
        num_assets: number of assets
        deg: data polynomial degree
        noise_level: level of data random noise
        seed: random seed (default 135 for reproducibility)

    Returns:
        tuple: covariance matrix (np.ndarray), data features (np.ndarray), mean returns (np.ndarray)
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
    # random matrix parameters
    B = rnd.binomial(1, 0.5, (m, p))
    L = rnd.uniform(-2.5e-3 * noise_level, 2.5e-3 * noise_level, (m, p))
    # feature vectors
    x = rnd.normal(0, 1, (n, p))
    # signal
    r = (0.05 * (x @ B.T) / np.sqrt(p) + 0.1 ** (1 / deg)) ** deg
    # noise: (f, eps) per row drawn together to match the per-i loop's RNG order
    fe = rnd.randn(n, p + m)
    F, E = fe[:, :p], fe[:, p:]
    r += F @ L.T + 0.01 * noise_level * E
    # covariance
    cov = L @ L.T + (1e-2 * noise_level) ** 2 * np.eye(m)
    return cov, x, r.astype(np.float32)
