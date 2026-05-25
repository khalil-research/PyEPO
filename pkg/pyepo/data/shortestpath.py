#!/usr/bin/env python
"""
Synthetic data for shortest path problem
"""

from __future__ import annotations

import numpy as np


def genData(
    num_data: int,
    num_features: int,
    grid: tuple[int, int],
    deg: int = 1,
    noise_width: float = 0,
    seed: int = 135,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic feature-cost pairs for the shortest path problem.

    Features are sampled from a standard multivariate Gaussian
    :math:`\\mathcal{N}(0, \\mathbf{I})`. A random Bernoulli(0.5) matrix
    :math:`\\mathcal{B}` maps each feature vector into the edge-cost
    coefficients via a polynomial of degree ``deg``, scaled by multiplicative
    uniform noise of half-width ``noise_width``.

    Args:
        num_data: number of data points
        num_features: dimension of features
        grid: size of grid network
        deg: polynomial degree of the feature-to-cost mapping
        noise_width: half-width of the multiplicative uniform noise
        seed: random seed (default 135 for reproducibility)

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
    # dimension of the cost vector
    d = (grid[0] - 1) * grid[1] + (grid[1] - 1) * grid[0]
    # random matrix parameter B
    B = rnd.binomial(1, 0.5, (d, p))
    # feature vectors
    x = rnd.normal(0, 1, (n, p))
    # cost vectors (vectorized)
    c = (x @ B.T / np.sqrt(p) + 3) ** deg + 1
    # rescale
    c /= 3.5**deg
    # noise
    epsilon = rnd.uniform(1 - noise_width, 1 + noise_width, (n, d))
    c *= epsilon

    return x, c.astype(np.float32)
