#!/usr/bin/env python
"""
Cone-aligned vector estimation (CaVE) loss for binary linear programs
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pyepo import EPO
from pyepo.func.cave import _HAS_CLARABEL, _project_one
from pyepo.func.jax.abcmodule import optModule


class coneAlignedCosine(optModule):
    """
    Cone-Aligned Vector Estimation (CaVE) loss for binary linear programs.

    Projects the sense-flipped predicted cost onto the polyhedral cone spanned
    by the binding-constraint normals at the true optimal vertex (a Clarabel
    QP) and minimizes :math:`1 - \\cos(-\\hat{\\mathbf{c}}, \\mathrm{proj})`.
    The projection is detached, so the gradient flows only through the cosine.

    Reference: Tang & Khalil (2024)
    `<https://link.springer.com/chapter/10.1007/978-3-031-60599-4_12>`_
    """

    def __init__(
        self, optmodel, max_iter=3, solve_ratio=1.0, inner_ratio=0.2, processes=1, reduction="mean"
    ):
        """
        Args:
            optmodel: a PyEPO optimization model (binary LP)
            max_iter: Clarabel iteration cap (3 = paper's CaVE+ preset)
            solve_ratio: per-batch QP-vs-heuristic probability (v1: must be 1.0)
            inner_ratio: heuristic-branch weight (unused in v1)
            processes: number of solver processes (1 = single-core)
            reduction: reduction applied to the batch loss ("mean", "sum", "none")
        """
        super().__init__(optmodel, processes, solve_ratio=1.0, reduction=reduction)
        if not _HAS_CLARABEL:
            raise ImportError(
                "clarabel is required for the CaVE cone projection. "
                "Install with `pip install clarabel`."
            )
        if solve_ratio != 1.0:
            raise NotImplementedError(
                "CaVE Hybrid (solve_ratio < 1) not yet supported in pyepo.func.jax"
            )
        self.max_iter = int(max_iter)
        self.inner_ratio = float(inner_ratio)
        # sense-aware sign (constant once optmodel is fixed)
        self._sign = -1.0 if optmodel.modelSense == EPO.MINIMIZE else 1.0

    def forward(self, pred_cost, tight_ctrs):
        """
        Forward pass
        """
        signed_cost = self._sign * pred_cost
        # stop the gradient into the projection
        proj = _clarabel_project(jax.lax.stop_gradient(signed_cost), tight_ctrs, self.max_iter)
        # cosine distance with a per-norm eps clamp
        den = jnp.maximum(jnp.linalg.norm(signed_cost, axis=1), 1e-8) * jnp.maximum(
            jnp.linalg.norm(proj, axis=1), 1e-8
        )
        loss = 1.0 - jnp.sum(signed_cost * proj, axis=1) / den
        return self._reduce(loss)


def _clarabel_project(signed_cost, tight_ctrs, max_iter):
    """Project each instance's signed cost onto its tight-constraint cone via Clarabel."""
    b, n = signed_cost.shape

    def _np_proj(sc, ct):
        projs = [
            _project_one(np.asarray(cp, np.float64), np.asarray(c, np.float64), max_iter)
            for cp, c in zip(sc, ct)
        ]
        return np.stack(projs).astype(np.float32)

    out = jax.ShapeDtypeStruct((b, n), jnp.float32)
    return jax.pure_callback(_np_proj, out, signed_cost, tight_ctrs)


# acronym aliases
CaVE = coneAlignedCosine
