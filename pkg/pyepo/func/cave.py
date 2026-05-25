#!/usr/bin/env python
"""
Cone-aligned vector estimation (CaVE) loss for binary linear programs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from pyepo import EPO
from pyepo.func.abcmodule import optModule

if TYPE_CHECKING:
    from pyepo.func.abcmodule import Reduction
    from pyepo.model.opt import optModel


class coneAlignedCosine(optModule):
    """
    An autograd module for the CaVE loss for binary linear programs. Projects
    the sense-flipped predicted cost onto the polyhedral cone spanned by the
    binding-constraint normals at the optimal vertex via batched dense
    Nesterov APGD on GPU/CPU, then minimizes ``1 - cos(pred, proj)``.

    Defaults reproduce the **CaVE+** preset from the paper (truncated APGD,
    ``max_iters=20``, no convergence check), which lands strictly inside the
    cone and avoids the boundary-stagnation that pure-exact APGD can hit. To
    approach the original **CaVE** preset, set a non-trivial tolerance
    (e.g. ``tol_grad=1e-4``) and remove the iteration cap (``max_iters=None``);
    in practice this is slower without a regret improvement, so the truncated
    default is recommended.

    Reference: <https://link.springer.com/chapter/10.1007/978-3-031-60599-4_12>
    """

    def __init__(
        self,
        optmodel: optModel,
        max_iters: int | None = 20,
        tol_grad: float | None = None,
        processes: int = 1,
        reduction: Reduction = "mean",
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            max_iters: APGD iteration cap. Default ``20`` (CaVE+ truncated);
                set to ``None`` to defer to a 10k safety bound for closer-to-exact behavior.
            tol_grad: L-inf tolerance on the projected gradient. Default ``None``
                skips the convergence check (pure fixed-iter, no sync per chunk).
                Set a non-trivial value (e.g. ``1e-4``) to converge to the cone boundary.
            processes: number of processors, 1 for single-core, 0 for all of cores
            reduction: the reduction to apply to the output
        """
        super().__init__(optmodel, processes, solve_ratio=1.0, reduction=reduction)
        self.tol_grad = tol_grad
        self.max_iters = max_iters

    def forward(
        self, pred_cost: torch.Tensor, tight_ctrs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass
        """
        # sense-aware sign
        if self.optmodel.modelSense == EPO.MINIMIZE:
            sign = -1.0
        elif self.optmodel.modelSense == EPO.MAXIMIZE:
            sign = 1.0
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        signed_cost = sign * pred_cost
        # fixed projection target
        with torch.no_grad():
            proj, _ = _apgd_project(
                tight_ctrs, signed_cost,
                tol_grad=self.tol_grad, max_iters=self.max_iters,
            )
            proj = proj / proj.norm(dim=1, keepdim=True).clamp(min=1e-8)
        loss = 1.0 - F.cosine_similarity(signed_cost, proj, dim=1)
        return self._reduce(loss)


def _apgd_project(
    tight_ctrs: torch.Tensor,
    signed_cost: torch.Tensor,
    tol_grad: float | None = 1e-4,
    max_iters: int | None = None,
    check_frequency: int = 200,
    power_iter: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A function to batch-project onto the binding-constraint cone via Nesterov APGD
    """
    # iteration cap
    cap = max_iters if max_iters is not None else 10_000
    # pre-transpose
    A = tight_ctrs.contiguous()
    AT = A.transpose(1, 2).contiguous()
    # per-instance step size
    step_size = _apgd_step_size(A, AT, power_iter)
    # Nesterov momentum schedule
    ks = torch.arange(cap, device=A.device, dtype=A.dtype)
    momenta = (ks - 2.0) / (ks + 1.0)
    momenta[0] = 0.0
    # cold-start state
    B, m, _ = A.shape
    x_curr = torch.zeros(B, m, device=A.device, dtype=A.dtype)
    x_prev = x_curr.clone()
    # chunked compiled loop with outer convergence check
    K = check_frequency
    for k_start in range(0, cap, K):
        K_actual = min(K, cap - k_start)
        # compiled chunk
        x_curr, x_prev = _apgd_iterate(
            A, AT, signed_cost, step_size, x_curr, x_prev, momenta[k_start:k_start + K_actual],
        )
        # clone breaks cudagraphs output-reuse aliasing
        x_curr = x_curr.clone()
        x_prev = x_prev.clone()
        # tol_grad=None: skip convergence check
        if tol_grad is None:
            continue
        # projected-gradient L-inf norm
        res = torch.bmm(AT, x_curr.unsqueeze(-1)).squeeze(-1) - signed_cost
        grad = torch.bmm(A, res.unsqueeze(-1)).squeeze(-1)
        active = x_curr > 0
        proj_grad = torch.where(active, grad.abs(), torch.clamp(-grad, min=0))
        # convergence check
        if proj_grad.max().item() <= tol_grad:
            break
    # final projection
    proj = torch.bmm(AT, x_curr.unsqueeze(-1)).squeeze(-1)
    # squared residual
    rnorm = (signed_cost - proj).pow(2).sum(dim=1)
    return proj, rnorm


def _apgd_step_size(
    A: torch.Tensor, AT: torch.Tensor, n_iter: int,
) -> torch.Tensor:
    """A function to estimate per-instance 1/lambda_max(A A^T) via power iteration"""
    B, m, _ = A.shape
    # deterministic unit-norm init for reproducibility
    v = torch.full((B, m), 1.0 / (m ** 0.5), device=A.device, dtype=A.dtype)
    # power iteration: v <- (A A^T) v
    for _ in range(n_iter):
        u = torch.bmm(AT, v.unsqueeze(-1)).squeeze(-1)
        v = torch.bmm(A, u.unsqueeze(-1)).squeeze(-1)
        # re-normalize
        v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-8)
    # Rayleigh quotient lambda = ||A^T v||^2
    u = torch.bmm(AT, v.unsqueeze(-1)).squeeze(-1)
    lam = (u * u).sum(dim=1)
    # invert for step size
    return (1.0 / lam.clamp(min=1e-8)).view(-1, 1)


@torch.compile(mode="reduce-overhead", dynamic=False)
def _apgd_iterate(
    A: torch.Tensor,
    AT: torch.Tensor,
    y: torch.Tensor,
    step_size: torch.Tensor,
    x_curr: torch.Tensor,
    x_prev: torch.Tensor,
    momenta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A fused chunk of Nesterov-accelerated PGD iterations"""
    n_iter = momenta.shape[0]
    for i in range(n_iter):
        # Nesterov extrapolation
        momentum = momenta[i]
        y_k = x_curr + momentum * (x_curr - x_prev)
        # residual A^T y_k - y
        res = torch.bmm(AT, y_k.unsqueeze(-1)).squeeze(-1) - y
        # gradient A (A^T y_k - y)
        grad = torch.bmm(A, res.unsqueeze(-1)).squeeze(-1)
        # save previous iterate
        x_prev = x_curr
        # projected gradient step onto x >= 0
        x_curr = torch.clamp(y_k - step_size * grad, min=0.0)
    return x_curr, x_prev
