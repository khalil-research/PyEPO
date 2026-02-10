#!/usr/bin/env python
# coding: utf-8
"""
Davis-Yin Splitting optimization function
"""

import numpy as np
import torch
from torch import nn


class dysOpt(nn.Module):
    """
    An autograd module for differentiable optimization using Davis-Yin
    Splitting with Jacobian-Free Backpropagation (DYS-Net), which yields an
    approximate optimal solution and derives a gradient via one-step unrolling.

    For DYS-Net, the objective function is linear and constraints are known and
    fixed, but the cost vector needs to be predicted from contextual data. The
    module replaces the combinatorial solver with an iterative three-operator
    splitting algorithm on a quadratically regularized continuous relaxation,
    and differentiates through the fixed point using one-step JFB.

    Thus, it allows us to design an algorithm based on stochastic gradient
    descent.

    Reference: <https://arxiv.org/abs/2301.13395>
    """

    def __init__(self, A, b, l, u, alpha=0.05, max_iter=1000, tol=1e-2, minimize=True):
        """
        Args:
            A (np.ndarray): equality constraint matrix (m x n), for Ax = b
            b (np.ndarray): equality constraint RHS vector (m,)
            l (np.ndarray): lower bounds on variables (n,)
            u (np.ndarray): upper bounds on variables (n,)
            alpha (float): step size / QP regularization strength
            max_iter (int): maximum number of DYS fixed-point iterations
            tol (float): convergence tolerance for fixed-point iteration
            minimize (bool): True for minimization, False for maximization
        """
        super().__init__()
        # convert inputs
        A = np.asarray(A, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        l = np.asarray(l, dtype=np.float32)
        u = np.asarray(u, dtype=np.float32)
        if A.ndim != 2:
            raise ValueError("A must be a 2D matrix, got shape {}".format(A.shape))
        if b.shape[0] != A.shape[0]:
            raise ValueError(
                "b has length {} but A has {} rows.".format(b.shape[0], A.shape[0]))
        n = A.shape[1]
        if l.shape[0] != n or u.shape[0] != n:
            raise ValueError(
                "l and u must have length {}, got {} and {}.".format(
                    n, l.shape[0], u.shape[0]))
        # step size
        if alpha <= 0:
            raise ValueError("alpha must be positive, got {}.".format(alpha))
        self.alpha = alpha
        # max iterations
        self.max_iter = max_iter
        # convergence tolerance
        self.tol = tol
        # number of variables
        self.n = n
        # minimize or maximize
        self.minimize = minimize
        # constraint data
        self.register_buffer("_A", torch.tensor(A))
        self.register_buffer("_b", torch.tensor(b))
        self.register_buffer("_lb", torch.tensor(l))
        self.register_buffer("_ub", torch.tensor(u))
        # precompute SVD for equality projection
        if A.shape[0] > 0:
            A_t = torch.tensor(A)
            U, s, VT = torch.linalg.svd(A_t, full_matrices=False)
            s_inv = torch.where(s >= 1e-6, 1.0 / s, torch.zeros_like(s))
            self.register_buffer("_V", VT.T)       # (n, r)
            self.register_buffer("_UT", U.T)        # (r, m)
            self.register_buffer("_s_inv", s_inv)   # (r,)
        else:
            # no equality constraints
            self.register_buffer("_V", torch.zeros(self.n, 0))
            self.register_buffer("_UT", torch.zeros(0, 0))
            self.register_buffer("_s_inv", torch.zeros(0))

    def _project_box(self, x):
        """
        Project onto box constraints [l, u]
        """
        return torch.clamp(x, min=self._lb, max=self._ub)

    def _project_equality(self, z):
        """
        Project onto equality constraint set {x: Ax = b} via SVD
        """
        if self._A.shape[0] == 0:
            return z
        # residual
        residual = self._A @ z.T - self._b.unsqueeze(1)
        # correction
        correction = self._V @ (self._s_inv.unsqueeze(1) * (self._UT @ residual))
        return z - correction.T

    def _grad_obj(self, z, cost):
        """
        Gradient of regularized objective: c + alpha * clamp(z, l, u)
        """
        return cost + self.alpha * self._project_box(z)

    def _dys_step(self, z, cost):
        """
        One Davis-Yin splitting iteration
        """
        x = self._project_box(z)
        y = self._project_equality(2.0 * x - z - self.alpha * self._grad_obj(z, cost))
        return z - x + y

    def _run_to_convergence(self, cost):
        """
        Run DYS iterations until convergence (no gradients)
        """
        batch_size = cost.shape[0]
        # init within bounds
        lb_safe = torch.where(torch.isfinite(self._lb), self._lb, torch.zeros_like(self._lb))
        ub_safe = torch.where(torch.isfinite(self._ub), self._ub, torch.ones_like(self._ub))
        z = torch.rand(batch_size, self.n, device=cost.device, dtype=cost.dtype) \
            * (ub_safe - lb_safe) + lb_safe
        # fixed-point iteration
        for i in range(self.max_iter):
            z_prev = z
            z = self._dys_step(z, cost)
            # check convergence
            diff = torch.norm(z - z_prev, dim=1).max()
            if diff <= self.tol:
                break
        if i >= self.max_iter - 1:
            print("Warning: DYS reached max_iter ({}) without convergence.".format(
                self.max_iter))
        return z

    def forward(self, pred_cost):
        """
        Forward pass
        """
        # negate cost for maximization
        cost = pred_cost if self.minimize else -pred_cost
        # solve
        with torch.no_grad():
            z = self._run_to_convergence(cost.detach())
        # one-step JFB for gradients
        if self.training:
            z = self._dys_step(z.detach(), cost)
            sol = self._project_box(z)
        else:
            sol = self._project_box(z).detach()
        return sol
