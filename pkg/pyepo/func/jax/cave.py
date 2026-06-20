#!/usr/bin/env python
"""
Cone-aligned vector estimation (CaVE) loss for binary linear programs
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from pyepo.func._common import is_minimize, validate_positive_int, validate_probability
from pyepo.func.cave import _HAS_CLARABEL, _project_one
from pyepo.func.jax.abcmodule import optModule
from pyepo.func.jax.utils import _concretizable


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
        validate_positive_int(max_iter, "max_iter")
        validate_probability(solve_ratio, "solve_ratio")
        validate_probability(inner_ratio, "inner_ratio")
        if not _HAS_CLARABEL:
            raise ImportError(
                "clarabel is required for the CaVE cone projection. "
                "Install with `pip install clarabel`."
            )
        # CaVE's ratio gates QP-vs-heuristic, not solution-pool caching.
        super().__init__(optmodel, processes, solve_ratio=1.0, reduction=reduction)
        self.max_iter = max_iter
        # ours gates the QP-vs-heuristic branch
        self.solve_ratio = float(solve_ratio)
        self.inner_ratio = float(inner_ratio)
        # sense-aware sign (constant once optmodel is fixed)
        self._sign = -1.0 if is_minimize(optmodel.modelSense) else 1.0

    def forward(self, pred_cost, tight_ctrs):
        """
        Forward pass
        """
        # labels carry no gradient (torch parity: detached projection target)
        tight_ctrs = jax.lax.stop_gradient(tight_ctrs)
        signed_cost = self._sign * pred_cost
        sc = jax.lax.stop_gradient(signed_cost)
        # the per-batch coin is host randomness; jit would freeze one branch
        if 0.0 < self.solve_ratio < 1.0 and not _concretizable(pred_cost):
            raise RuntimeError(
                "The CaVE Hybrid solve-vs-heuristic coin is not supported under jax.jit "
                "(or other abstract tracing); run eagerly or use solve_ratio 0 or 1."
            )
        # per-batch coin flip: QP projection or cheap heuristic
        if self._branch_rng.uniform() <= self.solve_ratio:
            proj = _clarabel_project_batch(sc, tight_ctrs, self.max_iter, self.pool)
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
    """Per-instance average of unit-normalized binding-constraint normals (zero-padded rows skipped)."""
    norms = jnp.linalg.norm(tight_ctrs, axis=2, keepdims=True)
    valid = (norms > 1e-7).astype(tight_ctrs.dtype)
    # unit-normalize valid rows, then average over them
    unit = tight_ctrs / jnp.maximum(norms, 1e-8) * valid
    n_valid = jnp.maximum(valid.sum(axis=1), 1.0)
    return unit.sum(axis=1) / n_valid


def _clarabel_project_batch(signed_cost, tight_ctrs, max_iter, pool=None):
    """
    A function to project each instance's signed cost onto its tight-constraint cone via Clarabel
    """
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
