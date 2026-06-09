#!/usr/bin/env python
"""
Cone-aligned vector estimation (CaVE) loss for binary linear programs
"""

from __future__ import annotations

from functools import partial

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
            solve_ratio: per-batch QP-vs-heuristic probability (< 1 activates CaVE Hybrid)
            inner_ratio: weight on the average binding-constraint normal in the heuristic branch
            processes: number of solver processes (1 = single-core, 0 = all cores)
            reduction: reduction applied to the batch loss ("mean", "sum", "none")
        """
        super().__init__(optmodel, processes, solve_ratio=1.0, reduction=reduction)
        if not _HAS_CLARABEL:
            raise ImportError(
                "clarabel is required for the CaVE cone projection. "
                "Install with `pip install clarabel`."
            )
        if not 0.0 <= solve_ratio <= 1.0:
            raise ValueError(f"Invalid solve_ratio {solve_ratio}; must be in [0, 1].")
        if not 0.0 <= inner_ratio <= 1.0:
            raise ValueError(f"Invalid inner_ratio {inner_ratio}; must be in [0, 1].")
        self.max_iter = int(max_iter)
        # ours gates the QP-vs-heuristic branch
        self.solve_ratio = float(solve_ratio)
        self.inner_ratio = float(inner_ratio)
        # sense-aware sign (constant once optmodel is fixed)
        self._sign = -1.0 if optmodel.modelSense == EPO.MINIMIZE else 1.0

    def forward(self, pred_cost, tight_ctrs):
        """
        Forward pass
        """
        signed_cost = self._sign * pred_cost
        sc = jax.lax.stop_gradient(signed_cost)
        # per-batch coin flip: QP projection or cheap heuristic
        if self._branch_rng.uniform() <= self.solve_ratio:
            proj = _clarabel_project(sc, tight_ctrs, self.max_iter, self.pool)
        else:
            # blend normalized pred with the average binding-constraint normal
            pred_n = sc / jnp.maximum(jnp.linalg.norm(sc, axis=1, keepdims=True), 1e-8)
            avg = _average_ctrs(tight_ctrs)
            proj = (1.0 - self.inner_ratio) * pred_n + self.inner_ratio * avg
        # cosine distance with a per-norm eps clamp
        den = jnp.maximum(jnp.linalg.norm(signed_cost, axis=1), 1e-8) * jnp.maximum(
            jnp.linalg.norm(proj, axis=1), 1e-8
        )
        loss = 1.0 - jnp.sum(signed_cost * proj, axis=1) / den
        return self._reduce(loss)


def _average_ctrs(tight_ctrs):
    """
    A function to average the unit-normalized binding-constraint normals per instance
    """
    norms = jnp.linalg.norm(tight_ctrs, axis=2, keepdims=True)
    valid = (norms > 1e-7).astype(tight_ctrs.dtype)
    # unit-normalize valid rows, then average over them
    unit = tight_ctrs / jnp.maximum(norms, 1e-8) * valid
    n_valid = jnp.maximum(valid.sum(axis=1), 1.0)
    return unit.sum(axis=1) / n_valid


def _clarabel_project(signed_cost, tight_ctrs, max_iter, pool=None):
    """Project each instance's signed cost onto its tight-constraint cone via Clarabel."""
    b, n = signed_cost.shape

    def _np_proj(sc, ct):
        cp_np = np.asarray(sc, np.float64)
        ctr_np = np.asarray(ct, np.float64)
        if pool is None:
            projs = [_project_one(cp, c, max_iter) for cp, c in zip(cp_np, ctr_np)]
        else:
            projs = pool.amap(partial(_project_one, max_iter=max_iter), cp_np, ctr_np).get()
        return np.stack(projs).astype(np.float32)

    out = jax.ShapeDtypeStruct((b, n), jnp.float32)
    return jax.pure_callback(_np_proj, out, signed_cost, tight_ctrs)


# acronym aliases
CaVE = coneAlignedCosine
