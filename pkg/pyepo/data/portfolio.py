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
    Generate synthetic feature-cost pairs for portfolio optimization.

    Returns the expected returns :math:`\\mathbf{r}` (the per-instance cost
    vectors) and a single shared covariance matrix :math:`\\mathbf{\\Sigma}`
    used in the risk constraint of the predefined portfolio model. The mean
    returns follow a factor-model structure
    :math:`\\mathbf{r}_i = \\bar{\\mathbf{r}}_i + \\mathbf{L}\\mathbf{f}
    + 0.01 \\tau \\boldsymbol{\\epsilon}`, where the factor loadings
    :math:`\\mathbf{L}` and residual noise are both scaled by ``noise_level``
    (:math:`\\tau`). Unlike the other generators in ``pyepo.data``, portfolio
    noise is controlled by ``noise_level`` rather than ``noise_width``.

    Args:
        num_data: number of data points
        num_features: dimension of features
        num_assets: number of assets
        deg: polynomial degree of the feature-to-return mapping
        noise_level: scales factor loadings L and residual noise (tau)
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
    # factor f and residual eps drawn in one call
    fe = rnd.randn(n, p + m)
    F, E = fe[:, :p], fe[:, p:]
    r += F @ L.T + 0.01 * noise_level * E
    # covariance
    cov = L @ L.T + (1e-2 * noise_level) ** 2 * np.eye(m)
    return cov, x, r.astype(np.float32)
