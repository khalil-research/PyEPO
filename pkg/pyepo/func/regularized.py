#!/usr/bin/env python
"""
Regularized differentiable optimization function
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch.autograd import Function

from pyepo import EPO
from pyepo.func.abcmodule import optModule
from pyepo.func.utils import _solve_or_cache

if TYPE_CHECKING:
    from pyepo.data.dataset import optDataset
    from pyepo.func.abcmodule import Reduction
    from pyepo.model.opt import optModel


class regularizedFrankWolfeOpt(optModule):
    """
    An autograd module for a regularized differentiable optimizer, which yields
    a smooth combination of vertices on the convex hull of feasible solutions.

    For regularized Frank-Wolfe, the objective function is linear and constraints
    are known and fixed, but the cost vector needs to be predicted from contextual
    data.

    The custom backward pass applies implicit differentiation on the active set
    found by Frank-Wolfe iterations. Thus, it allows us to design an algorithm
    based on stochastic gradient descent.

    Reference: <https://arxiv.org/abs/2207.13513>
    """

    def __init__(
        self,
        optmodel: optModel,
        lambd: float = 1.0,
        max_iter: int = 20,
        tol: float = 1e-6,
        processes: int = 1,
        solve_ratio: float = 1.0,
        dataset: optDataset | None = None,
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            lambd: a hyperparameter for regularized Frank-Wolfe to control the L2 regularization strength
            max_iter: number of Frank-Wolfe iterations
            tol: per-instance Frank-Wolfe gap convergence tolerance
            processes: number of processors, 1 for single-core, 0 for all of cores
            solve_ratio: the ratio of new solutions computed during training
            dataset: the training data
        """
        super().__init__(optmodel, processes, solve_ratio, dataset=dataset)
        # regularization strength
        if lambd <= 0:
            raise ValueError("lambda is not positive.")
        self.lambd = float(lambd)
        # Frank-Wolfe iteration budget
        self.max_iter = max_iter
        # Frank-Wolfe gap tolerance
        self.tol = tol

    def forward(self, pred_cost: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        return cast("torch.Tensor", regularizedFrankWolfeOptFunc.apply(pred_cost, self))

    def compute_regularization(self, y: torch.Tensor) -> torch.Tensor:
        """
        L2 regularizer Omega(y) = (lambd / 2) ||y||^2 per instance
        """
        return 0.5 * self.lambd * (y ** 2).sum(dim=-1)

    def _frankWolfe(
        self, theta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched Frank-Wolfe for argmin_mu 1/2 ||mu - theta||^2 over conv(V).

        All batch instances iterate in lockstep so each step is a single batched
        LMO call. Converged instances (gap < tol) get gamma = 0 for the remaining
        iterations. The active-set buffer has fixed shape (max_iter + 1, batch, vars).
        """
        # device, dtype, sizes
        batch, num_vars = theta.shape
        device, dtype = theta.device, theta.dtype
        # sense sign: flip so optmodel.solve runs as argmax
        if self.optmodel.modelSense == EPO.MINIMIZE:
            sense_sign = -1.0
        elif self.optmodel.modelSense == EPO.MAXIMIZE:
            sense_sign = 1.0
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        # initial vertex
        v0, _ = _solve_or_cache(sense_sign * theta, self)
        v0 = v0.to(device=device, dtype=dtype)
        # active-set buffer
        vertices = torch.zeros((batch, self.max_iter + 1, num_vars), device=device, dtype=dtype)
        weights = torch.zeros((batch, self.max_iter + 1), device=device, dtype=dtype)
        vertex_norms = torch.zeros((batch, self.max_iter + 1), device=device, dtype=dtype)
        vertices[:, 0] = v0
        weights[:, 0] = 1.0
        vertex_norms[:, 0] = (v0 * v0).sum(dim=-1)
        mu = v0.clone()
        # Frank-Wolfe iterations
        alive = torch.ones(batch, device=device, dtype=torch.bool)
        # loop buffers
        v = torch.zeros_like(mu)
        gap = torch.zeros(batch, device=device, dtype=dtype)
        batch_idx = torch.arange(batch, device=device)
        for k in range(self.max_iter):
            # Frank-Wolfe direction
            grad = mu - theta
            v_alive, _ = _solve_or_cache(sense_sign * (theta[alive] - mu[alive]), self)
            v[alive] = v_alive.to(device=device, dtype=dtype)
            diff = mu - v
            # per-instance Frank-Wolfe gap
            gap[alive] = (grad[alive] * diff[alive]).sum(dim=-1)
            active_mask = alive & (gap >= self.tol)
            active = active_mask.to(dtype)
            # early break when all instances converged
            if not bool(active_mask.any()):
                return mu, vertices[:, :k + 1], weights[:, :k + 1]
            # exact line search for the quadratic
            denom = (diff * diff).sum(dim=-1).clamp(min=1e-12)
            gamma = (gap / denom).clamp(0.0, 1.0) * active
            # in-place update
            mu.sub_(gamma.unsqueeze(-1) * diff)
            # vertex dedup via cached norms + inner products
            v_norm_sq = (v * v).sum(dim=-1)
            inner = torch.einsum("bnv,bv->bn", vertices[:, :k + 1], v)
            dist_sq = vertex_norms[:, :k + 1] - 2 * inner + v_norm_sq.unsqueeze(-1)
            match_mask = dist_sq < 1e-6
            has_match = match_mask.any(dim=-1)
            match_idx = match_mask.to(dtype).argmax(dim=-1)
            # in-place weight shrink
            weights.mul_((1.0 - gamma).unsqueeze(-1))
            weights[batch_idx, match_idx] = weights[batch_idx, match_idx] + gamma * has_match.to(dtype)
            vertices[:, k + 1] = v
            vertex_norms[:, k + 1] = v_norm_sq
            weights[:, k + 1] = gamma * (~has_match).to(dtype)
            alive = active_mask
        return mu, vertices, weights


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
        if module.optmodel.modelSense == EPO.MINIMIZE:
            sign = -1.0
        elif module.optmodel.modelSense == EPO.MAXIMIZE:
            sign = 1.0
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        scale = sign / module.lambd
        theta = scale * cp
        # batched Frank-Wolfe
        mu, vertices, weights = module._frankWolfe(theta)
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
        s = (weights > 0).to(dtype)                                       # (batch, K+1)
        # active-only mean per instance
        n_active = s.sum(dim=-1, keepdim=True).clamp(min=1.0)             # (batch, 1)
        V_mean = (vertices * s.unsqueeze(-1)).sum(dim=-2) / n_active      # (batch, vars)
        # centered, with inactive rows kept at zero
        V_centered = (vertices - V_mean.unsqueeze(-2)) * s.unsqueeze(-1)
        # project grad_output onto row space of V_centered via (K+1, K+1) Gram matrix
        M = V_centered @ V_centered.transpose(-1, -2)            # (batch, K+1, K+1)
        h = V_centered @ grad_output.unsqueeze(-1)               # (batch, K+1, 1)
        M.diagonal(dim1=-2, dim2=-1).add_(1e-6)                  # ridge for rank-deficient M
        alpha = torch.linalg.solve(M, h)                         # (batch, K+1, 1)
        grad = (V_centered.transpose(-1, -2) @ alpha).squeeze(-1)
        # chain rule for pred_cost
        return scale * grad, None


class regularizedFrankWolfeFenchelYoung(optModule):
    """
    An autograd module for the Fenchel-Young loss paired with regularized
    Frank-Wolfe.

    For regularized Frank-Wolfe Fenchel-Young loss, the objective function is
    linear and constraints are known and fixed, but the cost vector needs to be
    predicted from contextual data.

    The custom backward pass returns the Danskin subgradient y_hat - w directly,
    skipping the implicit-differentiation chain through the Frank-Wolfe iterate.
    Thus, it allows us to design an algorithm based on stochastic gradient
    descent.

    Reference: <https://arxiv.org/abs/2207.13513>
    """

    def __init__(
        self,
        optmodel: optModel,
        lambd: float = 1.0,
        max_iter: int = 20,
        tol: float = 1e-6,
        processes: int = 1,
        solve_ratio: float = 1.0,
        reduction: Reduction = "mean",
        dataset: optDataset | None = None,
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            lambd: a hyperparameter for regularized Frank-Wolfe to control the L2 regularization strength
            max_iter: number of Frank-Wolfe iterations
            tol: per-instance Frank-Wolfe gap convergence tolerance
            processes: number of processors, 1 for single-core, 0 for all of cores
            solve_ratio: the ratio of new solutions computed during training
            reduction: the reduction to apply to the output
            dataset: the training data
        """
        super().__init__(optmodel, processes, solve_ratio, reduction=reduction, dataset=dataset)
        # regularization strength
        if lambd <= 0:
            raise ValueError("lambda is not positive.")
        self.lambd = float(lambd)
        # Frank-Wolfe iteration budget
        self.max_iter = max_iter
        # Frank-Wolfe gap tolerance
        self.tol = tol

    def forward(self, pred_cost: torch.Tensor, true_sol: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        loss = cast(
            "torch.Tensor", regularizedFrankWolfeFenchelYoungFunc.apply(pred_cost, true_sol, self)
        )
        return self._reduce(loss)

    def _frankWolfe(
        self, theta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched Frank-Wolfe for argmin_mu 1/2 ||mu - theta||^2 over conv(V).

        Returns the final iterate mu only; the active set is not needed for
        the Danskin subgradient.
        """
        # device, dtype
        device, dtype = theta.device, theta.dtype
        # sense sign: flip so optmodel.solve runs as argmax
        if self.optmodel.modelSense == EPO.MINIMIZE:
            sense_sign = -1.0
        elif self.optmodel.modelSense == EPO.MAXIMIZE:
            sense_sign = 1.0
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        # initial vertex
        v0, _ = _solve_or_cache(sense_sign * theta, self)
        mu = v0.to(device=device, dtype=dtype)
        # Frank-Wolfe iterations
        batch = theta.shape[0]
        alive = torch.ones(batch, device=device, dtype=torch.bool)
        # loop buffers
        v = torch.zeros_like(mu)
        gap = torch.zeros(batch, device=device, dtype=dtype)
        for _ in range(self.max_iter):
            # Frank-Wolfe direction
            grad = mu - theta
            v_alive, _ = _solve_or_cache(sense_sign * (theta[alive] - mu[alive]), self)
            v[alive] = v_alive.to(device=device, dtype=dtype)
            diff = mu - v
            # per-instance Frank-Wolfe gap
            gap[alive] = (grad[alive] * diff[alive]).sum(dim=-1)
            active_mask = alive & (gap >= self.tol)
            active = active_mask.to(dtype)
            # early break when all instances converged
            if not bool(active_mask.any()):
                break
            # exact line search for the quadratic
            denom = (diff * diff).sum(dim=-1).clamp(min=1e-12)
            gamma = (gap / denom).clamp(0.0, 1.0) * active
            # in-place update
            mu.sub_(gamma.unsqueeze(-1) * diff)
            alive = active_mask
        return mu


class regularizedFrankWolfeFenchelYoungFunc(Function):
    """
    An autograd function for regularized Frank-Wolfe Fenchel-Young loss
    """

    @staticmethod
    def forward(
        ctx,
        pred_cost: torch.Tensor,
        true_sol: torch.Tensor,
        module: regularizedFrankWolfeFenchelYoung,
    ) -> torch.Tensor:
        """
        Forward pass for regularized Frank-Wolfe Fenchel-Young loss

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
        if module.optmodel.modelSense == EPO.MINIMIZE:
            r_sol = module._frankWolfe(-cp / module.lambd)
            diff = w - r_sol
        elif module.optmodel.modelSense == EPO.MAXIMIZE:
            r_sol = module._frankWolfe(cp / module.lambd)
            diff = r_sol - w
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        # regularizers
        omega_w = 0.5 * module.lambd * (w ** 2).sum(dim=-1)
        omega_r = 0.5 * module.lambd * (r_sol ** 2).sum(dim=-1)
        # Fenchel-Young loss
        loss = (omega_w - omega_r) + torch.einsum("bi,bi->b", cp, diff)
        # save solutions
        ctx.save_for_backward(diff)
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """
        Backward pass for regularized Frank-Wolfe Fenchel-Young loss
        """
        grad, = ctx.saved_tensors
        grad_output = torch.unsqueeze(grad_output, dim=-1)
        return grad * grad_output, None, None
