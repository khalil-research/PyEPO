#!/usr/bin/env python
"""
Synthetic data for traveling salesman problem
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import distance


def genData(
    num_data: int,
    num_features: int,
    num_nodes: int,
    deg: int = 1,
    noise_width: float = 0,
    seed: int = 135,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic feature-cost pairs for the traveling salesperson problem.

    Edge costs combine a Euclidean component (node coordinates drawn from a
    mixture of a Gaussian :math:`\\mathcal{N}(0, \\mathbf{I})` and a uniform
    :math:`\\mathbf{U}(-2, 2)` distribution) with a feature-encoded component
    obtained by mapping the standard-Gaussian feature vector through a random
    Bernoulli :math:`\\times` uniform matrix :math:`\\mathcal{B}` and a polynomial
    of degree ``deg``, scaled by multiplicative noise of half-width ``noise_width``.

    Args:
        num_data: number of data points
        num_features: dimension of features
        num_nodes: number of nodes
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
    if noise_width < 0:
        raise ValueError(f"noise_width = {noise_width} should be non-negative.")
    # set seed
    rnd = np.random.RandomState(seed)
    # number of data points
    n = num_data
    # dimension of features
    p = num_features
    # number of nodes
    m = num_nodes
    # random coordinates
    coords = np.concatenate((rnd.uniform(-2, 2, (m // 2, 2)), rnd.normal(0, 1, (m - m // 2, 2))))
    # condensed pairwise distances (upper triangle, row-major)
    dist_upper = distance.pdist(coords, "euclidean")
    # random matrix parameter B
    B = rnd.binomial(1, 0.5, (m * (m - 1) // 2, p)) * rnd.uniform(-2, 2, (m * (m - 1) // 2, p))
    # feature vectors
    x = rnd.normal(0, 1, (n, p))
    # feature-based cost
    feature_cost = ((x @ B.T / np.sqrt(p) + 3) ** deg) / 3 ** (deg - 1)
    # noise
    noise = rnd.uniform(1 - noise_width, 1 + noise_width, (n, m * (m - 1) // 2))
    # broadcast base distances over data points
    c = dist_upper + feature_cost * noise
    # rounding
    c = np.around(c, decimals=4).astype(np.float32)
    return x, c
