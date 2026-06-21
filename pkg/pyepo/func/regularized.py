#!/usr/bin/env python
"""
Regularized differentiable optimization function
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch.autograd import Function

from pyepo.func._common import (
    is_minimize,
    validate_nonnegative,
    validate_positive,
    validate_positive_int,
)
from pyepo.func.abcmodule import optModule
from pyepo.func.utils import _solve_or_cache

if TYPE_CHECKING:
    from pyepo.data.dataset import optDataset
    from pyepo.func.abcmodule import Reduction
    from pyepo.model.opt import optModel


def _fw_free_slot(weights: torch.Tensor) -> torch.Tensor:
    """Slot index for a new active vertex: the first free slot, or the smallest-weight atom when the buffer is full."""
    free_mask = weights <= 1e-12
    return torch.where(
        free_mask.any(dim=-1),
        free_mask.to(weights.dtype).argmax(dim=-1),
        weights.argmin(dim=-1),
    )


@torch.no_grad()
def _away_step_frank_wolfe(
    module: optModule,
    theta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched away-step Frank-Wolfe for argmin_mu 1/2 ||mu - theta||^2 over conv(V).

    Each step also considers the away direction from the worst active vertex and
    drops it when its weight reaches zero, so the iterate reaches the optimal face
    and the returned active set is its true support.

    Returns:
        tuple: final iterate mu (batch, vars), active vertices (batch, width, vars),
        and their weights (batch, width)
    """
    batch, num_vars = theta.shape
    device, dtype = theta.device, theta.dtype
    # sense sign: flip so optmodel.solve runs as argmax
    sense_sign = -1.0 if is_minimize(module.optmodel.modelSense) else 1.0
    # bounded active-set buffer (support <= num_vars + 1 by Caratheodory)
    width = 2 * num_vars + 2
    vertices = torch.zeros((batch, width, num_vars), device=device, dtype=dtype)
    weights = torch.zeros((batch, width), device=device, dtype=dtype)
    vertex_norms = torch.zeros((batch, width), device=device, dtype=dtype)
    # initial vertex
    v0, _ = _solve_or_cache(sense_sign * theta, module)
    v0 = v0.to(device=device, dtype=dtype)
    vertices[:, 0] = v0
    weights[:, 0] = 1.0
    vertex_norms[:, 0] = (v0 * v0).sum(dim=-1)
    mu = v0.clone()
    batch_idx = torch.arange(batch, device=device)
    max_iter, tol = cast("int", module.max_iter), cast("float", module.tol)
    for _ in range(max_iter):
        grad = mu - theta
        # Frank-Wolfe vertex and gap, solved for every instance each step
        v, _ = _solve_or_cache(sense_sign * (theta - mu), module)
        v = v.to(device=device, dtype=dtype)
        gap_fw = (grad * (mu - v)).sum(dim=-1)
        # away vertex: active atom maximizing <grad, .>
        scores = torch.einsum("bwv,bv->bw", vertices, grad).masked_fill(weights <= 0, float("-inf"))
        away_idx = scores.argmax(dim=-1)
        v_away = vertices[batch_idx, away_idx]
        alpha_away = weights[batch_idx, away_idx]
        gap_away = (grad * (v_away - mu)).sum(dim=-1)
        # zero the step on converged instances per iteration (recoverable, not frozen)
        unconverged = gap_fw >= tol
        if not bool(unconverged.any()):
            break
        active = unconverged.to(dtype)
        # choose Frank-Wolfe vs away direction per instance
        use_fw = gap_fw >= gap_away
        direction = torch.where(use_fw.unsqueeze(-1), v - mu, mu - v_away)
        gap = torch.where(use_fw, gap_fw, gap_away)
        gamma_max = torch.where(
            use_fw, torch.ones_like(alpha_away), alpha_away / (1.0 - alpha_away).clamp(min=1e-12)
        )
        # exact line search for the quadratic, clamped to the step cap
        denom = (direction * direction).sum(dim=-1).clamp(min=1e-12)
        gamma = torch.minimum((gap / denom).clamp(min=0.0), gamma_max) * active
        mu = mu + gamma.unsqueeze(-1) * direction
        # shrink weights: Frank-Wolfe *(1 - gamma), away *(1 + gamma)
        weights = weights * torch.where(use_fw, 1.0 - gamma, 1.0 + gamma).unsqueeze(-1)
        gamma_fw = gamma * use_fw
        gamma_away = gamma * (~use_fw)
        # Frank-Wolfe add: dedup against active atoms, else fill a free slot
        v_norm_sq = (v * v).sum(dim=-1)
        inner = torch.einsum("bwv,bv->bw", vertices, v)
        dist_sq = vertex_norms - 2 * inner + v_norm_sq.unsqueeze(-1)
        match = (dist_sq < 1e-6) & (weights > 0)
        has_match = match.any(dim=-1)
        match_idx = match.to(dtype).argmax(dim=-1)
        free_idx = _fw_free_slot(weights)
        # zero-weight adds carry no mass and must not displace an atom
        add_new = (~has_match) & use_fw & (gamma_fw > 0)
        weights[batch_idx, match_idx] = weights[batch_idx, match_idx] + gamma_fw * (
            has_match & use_fw
        ).to(dtype)
        vertices[batch_idx, free_idx] = torch.where(
            add_new.unsqueeze(-1), v, vertices[batch_idx, free_idx]
        )
        vertex_norms[batch_idx, free_idx] = torch.where(
            add_new, v_norm_sq, vertex_norms[batch_idx, free_idx]
        )
        weights[batch_idx, free_idx] = torch.where(
            add_new, gamma_fw + weights[batch_idx, free_idx], weights[batch_idx, free_idx]
        )
        # away subtract, then clear FP residue so dropped atoms leave the active set
        weights[batch_idx, away_idx] = weights[batch_idx, away_idx] - gamma_away
        weights = weights.clamp(min=0.0)
        weights = torch.where(weights < 1e-12, torch.zeros_like(weights), weights)
    return mu, vertices, weights


class regularizedFrankWolfeOpt(optModule):
    """
    L2-Regularized Frank-Wolfe Optimizer (RFWO) -- differentiable smoothed solver.

    Adds an L2 regularizer :math:`\\tfrac{\\lambda}{2}\\|\\mathbf{w}\\|_2^2`
    to the linear objective and returns the regularized minimizer
    :math:`\\hat{\\mathbf{w}}_\\lambda(\\hat{\\mathbf{c}}) =
    \\arg\\min_{\\mathbf{w} \\in \\mathrm{conv}(\\mathcal{S})}
    \\hat{\\mathbf{c}}^\\top \\mathbf{w} + \\tfrac{\\lambda}{2}\\|\\mathbf{w}\\|_2^2`,
    which is unique, lies inside :math:`\\mathrm{conv}(\\mathcal{S})`, and
    varies continuously with :math:`\\hat{\\mathbf{c}}`. The program is
    solved by batched Frank-Wolfe iteration; the only oracle needed is the
    underlying linear solver (PyEPO's standard ``optModel.solve``).

    Returns a regularized solution -- not a loss. Pair with a user-defined
    task loss (MSE against :math:`\\mathbf{w}^*(\\mathbf{c})` matches the
    imitation setting in the paper), or use ``regularizedFrankWolfeFenchelYoung``
    for the loss-returning variant.

    Reference: Dalle et al. (2022) `<https://arxiv.org/abs/2207.13513>`_
    """

    def __init__(
        self,
        optmodel: optModel,
        lambd: float = 1.0,
        max_iter: int = 10000,
        tol: float = 1e-6,
        processes: int = 1,
        solve_ratio: float = 1.0,
        dataset: optDataset | None = None,
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            lambd: L2 regularization strength :math:`\\lambda`
            max_iter: Frank-Wolfe iteration cap
            tol: per-instance Frank-Wolfe gap convergence tolerance
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of linear minimization oracle calls solved exactly each step (1.0 = no caching)
            dataset: training dataset used to seed the linear minimization oracle pool when ``solve_ratio < 1``
        """
        validate_positive(lambd, "lambda")
        validate_positive_int(max_iter, "max_iter")
        validate_nonnegative(tol, "tol")
        super().__init__(optmodel, processes, solve_ratio, dataset=dataset)
        # regularization strength
        self.lambd = float(lambd)
        # Frank-Wolfe iteration budget
        self.max_iter = max_iter
        # Frank-Wolfe gap tolerance
        self.tol = tol

    def forward(self, pred_cost: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # lift cost to the full objective space (no-op without partial prediction)
        pred_cost = self.optmodel._fullCost(pred_cost)
        return cast("torch.Tensor", regularizedFrankWolfeOptFunc.apply(pred_cost, self))

    def compute_regularization(self, y: torch.Tensor) -> torch.Tensor:
        """
        L2 regularizer Omega(y) = (lambd / 2) ||y||^2 per instance
        """
        return 0.5 * self.lambd * (y**2).sum(dim=-1)

    @torch.no_grad()
    def _frank_wolfe(
        self,
        theta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Away-step Frank-Wolfe over conv(V); returns mu and its active set.
        """
        return _away_step_frank_wolfe(self, theta)


class regularizedFrankWolfeOptFunc(Function):
    """
    An autograd function for regularized Frank-Wolfe
    """

    @staticmethod
    def forward(
        ctx,
        pred_cost: torch.Tensor,
        module: regularizedFrankWolfeOpt,
    ) -> torch.Tensor:
        """
        Forward pass for regularized Frank-Wolfe

        Args:
            pred_cost: a batch of predicted values of the cost
            module: regularizedFrankWolfeOpt module

        Returns:
            torch.tensor: regularized solutions
        """
        # convert tensor
        cp = pred_cost.detach()
        # rescale by sense and lambd
        sign = -1.0 if is_minimize(module.optmodel.modelSense) else 1.0
        scale = sign / module.lambd
        theta = scale * cp
        # batched Frank-Wolfe
        mu, vertices, weights = module._frank_wolfe(theta)
        # save active set
        ctx.save_for_backward(vertices, weights)
        # add other objects to ctx
        ctx.scale = scale
        return mu

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """
        Backward pass for regularized Frank-Wolfe
        """
        vertices, weights = ctx.saved_tensors
        scale = ctx.scale
        # batched orthogonal projector onto each instance's active hull
        dtype = vertices.dtype
        s = (weights > 0).to(dtype)  # (batch, K+1)
        # active-only mean per instance
        n_active = s.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (batch, 1)
        V_mean = (vertices * s.unsqueeze(-1)).sum(dim=-2) / n_active  # (batch, vars)
        # centered, with inactive rows kept at zero
        V_centered = (vertices - V_mean.unsqueeze(-2)) * s.unsqueeze(-1)
        # project grad_output onto row space of V_centered via (K+1, K+1) Gram matrix
        M = V_centered @ V_centered.transpose(-1, -2)  # (batch, K+1, K+1)
        h = V_centered @ grad_output.unsqueeze(-1)  # (batch, K+1, 1)
        # rtol-style ridge scaled to M
        ridge = M.diagonal(dim1=-2, dim2=-1).amax(dim=-1, keepdim=True).clamp(min=1.0) * 1e-6
        M.diagonal(dim1=-2, dim2=-1).add_(ridge)  # ridge for rank-deficient M
        alpha = torch.linalg.solve(M, h)  # (batch, K+1, 1)
        grad = (V_centered.transpose(-1, -2) @ alpha).squeeze(-1)
        # chain rule for pred_cost
        return scale * grad, None


class regularizedFrankWolfeFenchelYoung(optModule):
    """
    L2-Regularized Frank-Wolfe with Fenchel-Young loss (RFY).

    Pairs the RFWO regularized solver with the Fenchel-Young loss of the L2
    regularizer, returning a convex scalar loss that compares the predicted
    cost :math:`\\hat{\\mathbf{c}}` to the true optimum
    :math:`\\mathbf{w}^*(\\mathbf{c})` directly -- no user-defined task loss
    needed. Specialized to :math:`\\Omega(\\mathbf{w}) =
    \\tfrac{\\lambda}{2}\\|\\mathbf{w}\\|_2^2`, the loss collapses to a
    transparent "compare regularizers + linear residual" form.

    By Danskin's theorem the gradient is the simple residual
    :math:`\\mathbf{w}^*(\\mathbf{c}) - \\hat{\\mathbf{w}}_\\lambda
    (\\hat{\\mathbf{c}})`, so the backward path skips implicit
    differentiation through the Frank-Wolfe iterate.

    Reference: Dalle et al. (2022) `<https://arxiv.org/abs/2207.13513>`_
    """

    def __init__(
        self,
        optmodel: optModel,
        lambd: float = 1.0,
        max_iter: int = 10000,
        tol: float = 1e-6,
        processes: int = 1,
        solve_ratio: float = 1.0,
        reduction: Reduction = "mean",
        dataset: optDataset | None = None,
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            lambd: L2 regularization strength :math:`\\lambda`
            max_iter: Frank-Wolfe iteration cap
            tol: per-instance Frank-Wolfe gap convergence tolerance
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of linear minimization oracle calls solved exactly each step (1.0 = no caching)
            reduction: reduction applied to the batch loss (``"mean"``, ``"sum"``, ``"none"``)
            dataset: training dataset used to seed the linear minimization oracle pool when ``solve_ratio < 1``
        """
        validate_positive(lambd, "lambda")
        validate_positive_int(max_iter, "max_iter")
        validate_nonnegative(tol, "tol")
        super().__init__(optmodel, processes, solve_ratio, reduction=reduction, dataset=dataset)
        # regularization strength
        self.lambd = float(lambd)
        # Frank-Wolfe iteration budget
        self.max_iter = max_iter
        # Frank-Wolfe gap tolerance
        self.tol = tol

    def forward(self, pred_cost: torch.Tensor, true_sol: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # lift cost to the full objective space (no-op without partial prediction)
        pred_cost = self.optmodel._fullCost(pred_cost)
        loss = cast(
            "torch.Tensor", regularizedFrankWolfeFenchelYoungFunc.apply(pred_cost, true_sol, self)
        )
        return self._reduce(loss)

    @torch.no_grad()
    def _frank_wolfe(
        self,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Away-step Frank-Wolfe over conv(V); returns mu (the active set is not
        needed for the Danskin subgradient).
        """
        mu, _, _ = _away_step_frank_wolfe(self, theta)
        return mu


class regularizedFrankWolfeFenchelYoungFunc(Function):
    """
    An autograd function for regularized Frank-Wolfe with Fenchel-Young loss
    """

    @staticmethod
    def forward(
        ctx,
        pred_cost: torch.Tensor,
        true_sol: torch.Tensor,
        module: regularizedFrankWolfeFenchelYoung,
    ) -> torch.Tensor:
        """
        Forward pass for regularized Frank-Wolfe with Fenchel-Young loss

        Args:
            pred_cost: a batch of predicted values of the cost
            true_sol: a batch of true optimal solutions
            module: regularizedFrankWolfeFenchelYoung module

        Returns:
            torch.tensor: Fenchel-Young loss
        """
        # convert tensor
        cp = pred_cost.detach()
        w = true_sol.detach()
        # batched Frank-Wolfe
        if is_minimize(module.optmodel.modelSense):
            r_sol = module._frank_wolfe(-cp / module.lambd)
            diff = w - r_sol
        else:
            r_sol = module._frank_wolfe(cp / module.lambd)
            diff = r_sol - w
        # regularizers
        omega_w = 0.5 * module.lambd * (w**2).sum(dim=-1)
        omega_r = 0.5 * module.lambd * (r_sol**2).sum(dim=-1)
        # Fenchel-Young loss
        loss = (omega_w - omega_r) + torch.einsum("bi,bi->b", cp, diff)
        # save solutions
        ctx.save_for_backward(diff)
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """
        Backward pass for regularized Frank-Wolfe with Fenchel-Young loss
        """
        (grad,) = ctx.saved_tensors
        grad_output = torch.unsqueeze(grad_output, dim=-1)
        return grad * grad_output, None, None


# acronym aliases
RFWO = regularizedFrankWolfeOpt
RFY = regularizedFrankWolfeFenchelYoung
