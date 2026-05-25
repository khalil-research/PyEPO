#!/usr/bin/env python
"""
Synthetic data for knapsack problem
"""

from __future__ import annotations

import numpy as np


def genData(
    num_data: int,
    num_features: int,
    num_items: int,
    dim: int = 1,
    deg: int = 1,
    noise_width: float = 0,
    seed: int = 135,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic feature-cost pairs for the multi-dimensional knapsack.

    Item weights are fixed across instances; only the value (cost) of each
    item depends on features. Features are sampled from a standard Gaussian
    :math:`\\mathcal{N}(0, \\mathbf{I})`, mapped through a random Bernoulli(0.5)
    matrix :math:`\\mathcal{B}` and a polynomial of degree ``deg``, then
    scaled by multiplicative uniform noise of half-width ``noise_width`` and
    rounded up to the nearest integer.

    Args:
        num_data: number of data points
        num_features: dimension of features
        num_items: number of items
        dim: dimension of multi-dimensional knapsack
        deg: polynomial degree of the feature-to-cost mapping
        noise_width: half-width of the multiplicative uniform noise
        seed: random state seed (default 135 for reproducibility)

    Returns:
       tuple: weights of items (np.ndarray), data features (np.ndarray), costs (np.ndarray)
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
    # dimension of problem
    d = dim
    # number of items
    m = num_items
    # weights of items
    weights = rnd.choice(range(300, 800), size=(d, m)) / 100
    # random matrix parameter B
    B = rnd.binomial(1, 0.5, (m, p))
    # feature vectors
    x = rnd.normal(0, 1, (n, p))
    # value of items (vectorized)
    c = (x @ B.T / np.sqrt(p) + 3) ** deg + 1
    # rescale
    c *= 5
    c /= 3.5**deg
    # noise
    epsilon = rnd.uniform(1 - noise_width, 1 + noise_width, (n, m))
    c *= epsilon
    # convert into int
    c = np.ceil(c).astype(np.float32)
    return weights, x, c
