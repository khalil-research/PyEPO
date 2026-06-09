#!/usr/bin/env python
"""
Abstract optimization model based on MPAX
"""

from __future__ import annotations

import dataclasses
import logging
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING

import torch

try:
    import jax
    from jax import numpy as jnp
    from mpax import create_lp, create_qp, raPDHG
    from mpax.termination import TerminationStatus

    _HAS_MPAX = True
except ImportError:
    _HAS_MPAX = False

from pyepo import EPO
from pyepo.model.opt import optModel

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self

logger = logging.getLogger(__name__)

# warn once across all solves if PDHG stops short of OPTIMAL
_warned_not_optimal = False


def _warn_if_not_optimal(status) -> None:
    """Warn once if any MPAX solve did not reach OPTIMAL termination."""
    global _warned_not_optimal
    if _warned_not_optimal:
        return
    flat = jnp.asarray(status).reshape(-1)
    if not bool(jnp.any(flat != int(TerminationStatus.OPTIMAL))):
        return
    _warned_not_optimal = True
    example = int(flat[jnp.argmax(flat != int(TerminationStatus.OPTIMAL))])
    logger.warning(
        "MPAX did not reach OPTIMAL termination for some instances (e.g. status=%s); "
        "the returned solution may be suboptimal or infeasible. Consider raising iteration_limit.",
        TerminationStatus(example).name,
    )


class optMpaxModel(optModel):
    """
    Abstract base class for MPAX-backed (JAX) linear / quadratic program models.

    MPAX is a JAX implementation of the PDHG (Primal-Dual Hybrid Gradient)
    first-order solver, designed for large-scale continuous programs that
    benefit from GPU acceleration and vmap-batched solving. Unlike the
    Gurobi / COPT / Pyomo / OR-Tools backends, an MPAX model has **no
    explicit solver model object** -- the constraint matrices and bounds
    *are* the model. Subclasses populate them inside ``_getModel`` and
    return ``(None, [])``::

        def _getModel(self):
            self.A = jnp.array(...)   # equality A x = b
            self.b = jnp.array(...)
            self.G = jnp.array(...)   # inequality G x >= h
            self.h = jnp.array(...)
            self.l = jnp.array(...)   # variable lower bound
            self.u = jnp.array(...)   # variable upper bound
            # optional: leave None for LP, set for convex QP
            self.Q = jnp.array(...)   # PSD; objective is 0.5 xᵀQx + cᵀx
            return None, []

    LP vs QP is selected automatically from ``self.Q``: ``None`` (default)
    keeps the LP code path via ``create_lp``, any other value routes
    through ``create_qp``. ``Q`` must be PSD; MPAX supports quadratic
    *objective only* -- constraints stay linear (this is a hard MPAX
    limit; e.g. a quadratic risk-budget constraint cannot be expressed).

    Objective sense follows ``self.modelSense`` (set by a problem-level base
    such as ``knapsackBase`` or directly in ``_getModel``; defaults to
    minimization). Dense vs sparse matrices can be toggled by overriding the
    class attribute ``use_sparse_matrix`` (default ``True``).

    A jitted single-instance solver and a ``vmap``-batched solver
    (``batch_optimize``) are pre-compiled on construction, so
    ``optDataset`` can solve every training instance in a single dispatch.

    Attributes:
        A (jnp.ndarray): equality-constraint matrix (Ax = b)
        b (jnp.ndarray): equality-constraint right-hand side
        G (jnp.ndarray): inequality-constraint matrix (Gx >= h)
        h (jnp.ndarray): inequality-constraint right-hand side
        l (jnp.ndarray): variable lower bounds
        u (jnp.ndarray): variable upper bounds
        Q (jnp.ndarray | None): PSD quadratic-objective matrix; ``None`` ⇒ LP
        use_sparse_matrix (bool): whether to use sparse matrices
    """

    use_sparse_matrix: bool = True
    # None ⇒ LP path; subclasses opt into QP by assigning self.Q in _getModel
    Q = None

    def __init__(self) -> None:
        if not _HAS_MPAX:
            raise ImportError("MPAX is not installed. Please install MPAX to use this feature.")
        super().__init__()
        # MAXIMIZE with PSD Q is non-convex; reject early
        if self.Q is not None and self.modelSense != EPO.MINIMIZE:
            raise ValueError(
                "MPAX QP path supports MINIMIZE sense only "
                "(MAXIMIZE with PSD Q is non-convex). "
                "Reformulate as MINIMIZE of -obj."
            )
        # init device
        self.device = None
        # cache JAX GPU device (None if CPU-only)
        self._gpu_device = next((d for d in jax.devices() if d.platform == "gpu"), None)
        self._has_jax_gpu = self._gpu_device is not None
        # JIT pre-compile (LP/QP dispatch on self.Q)
        self._rebuild_jit()

    def __repr__(self) -> str:
        return "optMpaxModel " + self.__class__.__name__

    def _rebuild_jit(self) -> None:
        """Rebuild solve functions; dispatches LP/QP from self.Q."""
        # LP path
        if self.Q is None:
            solve_fn = partial(
                self._jitted_solve_lp,
                A=self.A,
                b=self.b,
                G=self.G,
                h=self.h,
                l=self.l,
                u=self.u,
                use_sparse_matrix=self.use_sparse_matrix,
            )
        # QP path (see _jitted_solve_qp)
        else:
            n = self.A.shape[1]
            # placeholder c (replaced per call)
            qp_template = create_qp(
                self.Q,
                jnp.zeros(n, dtype=jnp.float32),
                self.A, self.b, self.G, self.h, self.l, self.u,
                use_sparse_matrix=self.use_sparse_matrix,
            )
            # Python bool keeps is_lp static under jit/vmap
            qp_template = dataclasses.replace(qp_template, is_lp=False)
            solve_fn = partial(self._jitted_solve_qp, qp_template=qp_template, Q=self.Q)
        # single-instance jit; batch path keeps jit outermost for one fused executable
        self.jitted_solve = jax.jit(solve_fn)
        self.batch_optimize = jax.jit(jax.vmap(solve_fn))

    @property
    def num_cost(self) -> int:
        """
        number of costs to be predicted
        """
        return self.A.shape[1]

    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        """
        A method to set the objective function

        Args:
            c: cost of objective function
        """
        # validate shape before assignment
        c_len = c.shape[-1] if hasattr(c, "shape") else len(c)
        if c_len != self.num_cost:
            raise ValueError("Size of cost vector does not match number of cost variables.")
        # check if c is a PyTorch tensor
        if isinstance(c, torch.Tensor):
            # move to cpu if JAX has no GPU support
            if not self._has_jax_gpu:
                c = c.cpu().detach()
            else:
                c = c.detach()
            # match float32 constraints; no-op (no copy) if already float32
            c = c.to(torch.float32)
            # convert PyTorch tensor to JAX array using DLPack
            self.c = jnp.from_dlpack(c)
            # keep solve on GPU even when cost came from a CPU tensor
            if self._gpu_device is not None:
                self.c = jax.device_put(self.c, self._gpu_device)
            # move constraints and bounds to device
            if self.device != self.c.device:
                self.device = self.c.device
                self.A = jax.device_put(self.A, self.device)
                self.b = jax.device_put(self.b, self.device)
                self.G = jax.device_put(self.G, self.device)
                self.h = jax.device_put(self.h, self.device)
                self.l = jax.device_put(self.l, self.device)
                self.u = jax.device_put(self.u, self.device)
                if self.Q is not None:
                    self.Q = jax.device_put(self.Q, self.device)
                # rebuild JIT for new device
                self._rebuild_jit()
        # c is already a NumPy array
        else:
            self.c = jnp.array(c, dtype=jnp.float32)
        # change sign for maximization
        if self.modelSense == EPO.MAXIMIZE:
            self.c = -self.c

    def solve(self) -> tuple[torch.Tensor, float]:
        """
        A method to solve the model

        Returns:
            tuple: optimal solution (torch.Tensor) and objective value (float)
        """
        # create lp model
        sol, obj, status = self.jitted_solve(self.c)
        _warn_if_not_optimal(status)
        # convert to torch
        sol = torch.from_dlpack(sol)
        if self.modelSense == EPO.MINIMIZE:
            obj = obj.item()
        elif self.modelSense == EPO.MAXIMIZE:
            obj = -obj.item()
        else:
            raise ValueError("Invalid modelSense.")
        return sol, obj

    @staticmethod
    def _jitted_solve_lp(c, A, b, G, h, l, u, use_sparse_matrix):
        """JIT-compiled LP solve (cᵀx)."""
        lp = create_lp(c, A, b, G, h, l, u, use_sparse_matrix=use_sparse_matrix)
        solver = raPDHG(
            eps_abs=1e-4, eps_rel=1e-4, verbose=False, iteration_limit=50_000,
        )
        result = solver.optimize(lp)
        obj = jnp.dot(c, result.primal_solution)
        return result.primal_solution, obj, result.termination_status

    @staticmethod
    def _jitted_solve_qp(c, qp_template, Q):
        """
        JIT-compiled QP solve (0.5 xᵀQx + cᵀx).

        ``qp_template`` must have ``is_lp = False`` (a Python bool, not a
        jnp scalar) so JAX's pytree flatten keeps it as static metadata;
        ``raPDHG.check_config`` then branches on a concrete value instead
        of a tracer. Solver instance is built inline (not closed over) to
        avoid cross-trace state leaks via raPDHG's internal mutation in
        ``select_initial_primal_weight``.
        """
        qp = dataclasses.replace(qp_template, objective_vector=c)
        solver = raPDHG(
            eps_abs=1e-4, eps_rel=1e-4, verbose=False, iteration_limit=50_000,
        )
        result = solver.optimize(qp)
        x = result.primal_solution
        obj = 0.5 * jnp.dot(x, Q @ x) + jnp.dot(c, x)
        return x, obj, result.termination_status

    def copy(self) -> Self:
        """
        A method to copy the model

        Returns:
            optModel: new copied model
        """
        # jax Device objects can't be deepcopied; stash and restore around the copy
        device, gpu_device = self.device, self._gpu_device
        self.device = None
        self._gpu_device = None
        # copy new model
        new_model = deepcopy(self)
        # restore device
        self.device, self._gpu_device = device, gpu_device
        new_model.device, new_model._gpu_device = device, gpu_device
        return new_model

    def addConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> Self:
        """
        A method to add a new constraint

        Args:
            coefs: coefficients of new constraint
            rhs: right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        if len(coefs) != self.num_cost:
            raise ValueError("Size of coef vector does not match number of cost variables.")
        # flip sign: PyEPO convention is `coefs · x <= rhs`, MPAX stores `G x >= h`
        coefs = -jnp.array(coefs, dtype=jnp.float32).reshape(1, -1)
        rhs = -float(rhs)
        # copy
        new_model = self.copy()
        # add constraint
        if new_model.G.shape[0] == 0:
            new_model.G = coefs
            new_model.h = jnp.array([rhs])
        else:
            new_model.G = jnp.vstack([new_model.G, coefs])
            new_model.h = jnp.append(new_model.h, rhs)
        # rebuild JIT with updated constraints
        new_model._rebuild_jit()
        return new_model
