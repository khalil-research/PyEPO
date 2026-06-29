#!/usr/bin/env python
"""
Cone-aligned vector estimation (CaVE) loss for binary linear programs
"""

from __future__ import annotations

from functools import cache, partial
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.nn import functional as F

from pyepo.func._common import is_minimize, validate_positive_int, validate_probability
from pyepo.func.abcmodule import optModule

try:
    import clarabel
    from scipy import sparse

    _HAS_CLARABEL = True
except ImportError:
    _HAS_CLARABEL = False

if TYPE_CHECKING:
    from pyepo.func.abcmodule import Reduction
    from pyepo.model.opt import optModel


class coneAlignedCosine(optModule):
    """
    Cone-Aligned Vector Estimation (CaVE) loss for binary linear programs.

    For each training instance, the sense-flipped predicted cost vector
    :math:`-\\hat{\\mathbf{c}}` is projected onto the polyhedral cone spanned
    by the binding-constraint normals at the true optimal vertex; the loss
    is :math:`1 - \\cos(-\\hat{\\mathbf{c}}, \\mathrm{proj})`. Because the
    supervision is the cone of binding normals (not the optimal solution
    itself), CaVE side-steps the zero-gradient pathology of solver layers
    without requiring a perturbation or solution pool. Defined for
    **binary linear programs** only.

    PyEPO uses Clarabel as the interior-point QP solver for the cone
    projection.

    .. note::
       The default ``max_iter=3`` is intentional — it is the **CaVE+**
       preset from the paper. Three IPM steps under-converge the QP on
       purpose so the projection stays interior to the cone, yielding a
       richer gradient than a fully converged boundary projection.
       Raising ``max_iter`` changes the loss behavior.

    For larger problems, set ``solve_ratio < 1`` to enable the **CaVE
    Hybrid** preset from the paper: each batch goes through the QP
    projection with probability ``solve_ratio`` and through a cheap
    heuristic (normalized predicted cost blended with the average
    binding-constraint normal) with probability ``1 - solve_ratio``,
    cutting the per-epoch cost without measurable regret loss.

    Training data must come from ``pyepo.data.dataset.optDatasetConstrs``
    (Gurobi-backed) and be batched with ``pyepo.data.dataset.optDataLoader``
    or a ``DataLoader`` using ``collate_tight_constraints``.

    Reference: Tang & Khalil (2024)
    `<https://link.springer.com/chapter/10.1007/978-3-031-60599-4_12>`_
    """

    def __init__(
        self,
        optmodel: optModel,
        max_iter: int = 3,
        solve_ratio: float = 1.0,
        inner_ratio: float = 0.2,
        processes: int = 1,
        reduction: Reduction = "mean",
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model (Gurobi-backed)
            max_iter: Clarabel iteration cap. Default ``3`` is the paper's
                CaVE+ preset; raising it changes the loss (see class note).
            solve_ratio: per-batch probability of running the QP projection.
                ``< 1`` activates the CaVE Hybrid heuristic branch.
            inner_ratio: weight on the average binding-constraint normal in
                the heuristic branch (only used when ``solve_ratio < 1``).
            processes: number of solver processes (1 = single-core, 0 = all cores)
            reduction: reduction applied to the batch loss (``"mean"``, ``"sum"``, ``"none"``)
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
        # override parent's solve_ratio: ours gates the QP-vs-heuristic branch
        self.solve_ratio = float(solve_ratio)
        self.inner_ratio = float(inner_ratio)
        # sense-aware sign (constant once optmodel is fixed)
        self._sign = -1.0 if is_minimize(optmodel.modelSense) else 1.0

    def forward(
        self,
        pred_cost: torch.Tensor,
        tight_ctrs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass
        """
        signed_cost = self._sign * pred_cost
        # tight_ctrs only consumed on CPU (Clarabel + avg both CPU-bound); avoid a GPU round-trip
        tight_ctrs_cpu = tight_ctrs.detach().cpu()
        # fixed projection target; gradient flows only through signed_cost below
        with torch.no_grad():
            # per-batch coin flip: QP branch or cheap heuristic branch
            if self._branch_rng.uniform() <= self.solve_ratio:
                # QP branch: truncated Clarabel projection inside the cone
                proj = _clarabel_project_batch(
                    signed_cost,
                    tight_ctrs_cpu,
                    max_iter=self.max_iter,
                    pool=self.pool,
                )
            else:
                # heuristic branch: blend normalized pred with avg binding-constraint normal
                pred_n = signed_cost / signed_cost.norm(dim=1, keepdim=True).clamp(min=1e-8)
                avg = _average_ctrs(tight_ctrs_cpu).to(signed_cost.device, signed_cost.dtype)
                proj = (1 - self.inner_ratio) * pred_n + self.inner_ratio * avg
        loss = F.cosine_similarity(signed_cost, proj, dim=1).neg().add(1.0)
        return self._reduce(loss)


def _average_ctrs(tight_ctrs: torch.Tensor) -> torch.Tensor:
    """Per-instance average of unit-normalized binding-constraint normals (zero-padded rows skipped)."""
    # unit-normalize each row
    norms = tight_ctrs.norm(dim=2, keepdim=True)
    valid = (norms > 1e-7).to(tight_ctrs.dtype)
    unit = tight_ctrs / norms.clamp(min=1e-8) * valid
    # average over valid rows
    n_valid = valid.sum(dim=1).clamp(min=1.0)
    return unit.sum(dim=1) / n_valid


@cache
def _neg_eye_csc(m: int):
    """Cached ``-I`` (CSC). Encodes ``lam >= 0`` as Clarabel ``A lam + s = b, s in cone`` with ``A = -I``, ``b = 0``."""
    return (-sparse.eye(m, format="csc")).tocsc()


@cache
def _clarabel_settings(max_iter: int):
    """Cached Clarabel settings; treat as immutable across calls."""
    s = clarabel.DefaultSettings()
    s.max_iter = int(max_iter)
    s.verbose = False
    # single-threaded for tiny per-instance QPs
    s.max_threads = 1
    # skip presolve / equilibrate
    s.presolve_enable = False
    s.equilibrate_enable = False
    return s


def _project_one(cp: np.ndarray, ctr: np.ndarray, max_iter: int) -> np.ndarray:
    """Project ``cp`` onto cone{lam @ ctr : lam >= 0} via one Clarabel solve."""
    # drop padded rows
    ctr = ctr[np.abs(ctr).sum(axis=1) > 1e-7]
    # no binding constraints
    if len(ctr) == 0:
        return cp.astype(np.float32)
    m = ctr.shape[0]
    # quadratic term
    P = sparse.triu(sparse.csc_matrix(2.0 * (ctr @ ctr.T))).tocsc()
    # linear term
    q = -2.0 * (ctr @ cp)
    # lam >= 0
    A = _neg_eye_csc(m)
    b = np.zeros(m)
    cones = [clarabel.NonnegativeConeT(m)]
    # build + solve
    solver = clarabel.DefaultSolver(P, q, A, b, cones, _clarabel_settings(max_iter))
    sol = solver.solve()
    # proj = lam @ ctr
    lam = np.asarray(sol.x)
    return (lam @ ctr).astype(np.float32)


def _clarabel_project_batch(
    signed_cost: torch.Tensor,
    tight_ctrs: torch.Tensor,
    max_iter: int,
    pool=None,
) -> torch.Tensor:
    """
    A function to project each instance's signed cost onto its tight-constraint cone via Clarabel
    """
    # device / dtype of the gradient-carrying input
    device, dtype = signed_cost.device, signed_cost.dtype
    # to float64 numpy (tight_ctrs already on CPU per forward)
    cp_np = signed_cost.detach().cpu().numpy().astype(np.float64)
    ctrs_np = tight_ctrs.numpy().astype(np.float64)
    # per-instance solve
    if pool is None:
        # single-core
        projs = [_project_one(cp, ctr, max_iter) for cp, ctr in zip(cp_np, ctrs_np)]
    else:
        # multi-core (pathos pool, pre-built per abcmodule.optModule)
        projs = pool.amap(partial(_project_one, max_iter=max_iter), cp_np, ctrs_np).get()
    # stack + back to torch
    return torch.as_tensor(np.stack(projs), dtype=dtype, device=device)


# acronym aliases
CaVE = coneAlignedCosine
