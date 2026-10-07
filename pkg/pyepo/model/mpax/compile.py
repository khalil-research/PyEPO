#!/usr/bin/env python
"""
MPAX (JAX PDHG) compiler for the PyEPO DSL.

``compiledMpaxProblem`` mixes the generic ``compiledBase`` with ``optMpaxModel``
to turn a finalized DSL ``Problem`` into MPAX standard-form matrices
(``min cᵀx + ½xᵀQx`` s.t. ``Ax = b``, ``Gx ≥ h``, ``l ≤ x ≤ u``) solved by the
JAX first-order solver. Unlike the other backends it **overrides** ``setObj`` /
``solve`` rather than using ``compiledBase``'s numpy hooks: the cost is kept as a
device tensor (DLPack) so vmap-batched GPU solving is preserved. MPAX is a
continuous LP / QP relaxation solver — integer / binary variables are relaxed to
their bounds, and quadratic *constraints* are not expressible.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy import sparse

try:
    import jax
    from jax import numpy as jnp
    from jax.experimental.sparse import BCOO
except ImportError:
    jax = None
    jnp = None

from pyepo import EPO
from pyepo.dsl.compiled import compiledBase
from pyepo.model._common import validate_objective_shape
from pyepo.model.mpax.mpaxmodel import _warn_if_not_optimal, optMpaxModel

logger = logging.getLogger(__name__)


def compileProblem(problem, **params) -> compiledMpaxProblem:
    """Instantiate the MPAX-compiled problem."""
    return compiledMpaxProblem(problem, params=params)


class compiledMpaxProblem(compiledBase, optMpaxModel):
    """
    MPAX-backed (JAX LP / QP) compiled DSL problem.
    """

    use_sparse_matrix = True

    def _getModel(self) -> tuple:
        # assemble MPAX standard-form matrices from the finalized IR
        prob = self.problem
        self.modelSense = prob.modelSense
        # warn on relaxed integrality
        if np.any(np.asarray(prob.var_type) != EPO.CONTINUOUS):
            logger.warning(
                "MPAX is a continuous solver; integer/binary variables are relaxed "
                "to their bounds and solutions may be fractional."
            )
        self._emit_constraints()
        self.l = jnp.asarray(
            np.where(np.isneginf(prob.var_lb), -np.inf, prob.var_lb).astype(np.float32)
        )
        self.u = jnp.asarray(
            np.where(np.isposinf(prob.var_ub), np.inf, prob.var_ub).astype(np.float32)
        )
        # quadratic objective (None ⇒ LP); Q = 2·obj_Q for MPAX's ½xᵀQx convention
        self.Q = self._matrix(2.0 * prob.obj_Q) if prob.obj_Q is not None else None
        return None, []

    def _emit_constraints(self):
        # split the IR constraints into equality (A x = b) and inequality (G x ≥ h) blocks
        n = self.problem.num_vars
        A_eq, b_eq, G, h = [], [], [], []
        for Q, A, sense, b in self.problem.constrs:
            if Q is not None:
                raise NotImplementedError(
                    "MPAX supports a quadratic objective only, not quadratic constraints."
                )
            A = A.astype(np.float32)
            b = np.asarray(b, dtype=np.float32).reshape(-1)
            if sense == "==":
                A_eq.append(A)
                b_eq.append(b)
            elif sense == "<=":
                G.append(-A)  # A x <= b  ->  -A x >= -b
                h.append(-b)
            else:
                G.append(A)  # A x >= b
                h.append(b)
        self.A = self._matrix(sparse.vstack(A_eq) if A_eq else sparse.coo_matrix((0, n)))
        self.b = jnp.asarray(np.concatenate(b_eq) if b_eq else np.zeros(0, np.float32))
        self.G = self._matrix(sparse.vstack(G) if G else sparse.coo_matrix((0, n)))
        self.h = jnp.asarray(np.concatenate(h) if h else np.zeros(0, np.float32))

    def _matrix(self, matrix):
        matrix = matrix.astype(np.float32).tocoo()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        if self.use_sparse_matrix:
            return BCOO.from_scipy_sparse(matrix)
        return jnp.asarray(matrix.toarray())

    def _apply_params(self):
        # MPAX (first-order PDHG) has no time-limit knob; accept `timelimit` and ignore it
        extra = {k: v for k, v in self.params.items() if k != "timelimit"}
        if extra:
            raise ValueError("MPAX backend does not accept solver params.")

    def setObj(self, c):
        """Set the objective from a predicted cost of length ``num_cost``, scattered onto the known fixed costs."""
        prob = self.problem
        validate_objective_shape(c, (prob.num_cost, prob.num_vars), allow_batch=True)
        n = c.shape[-1] if hasattr(c, "shape") else np.shape(c)[-1]
        # scatter onto fixed costs; an unambiguous full-length vector passes through
        if n == prob.num_cost:
            self._write_cost(c, is_full=False)
        else:
            self._write_cost(c, is_full=True)

    def _setFullObj(self, c):
        """Set the objective from full-space coefficients (length ``num_vars``), bypassing the predicted-cost scatter."""
        validate_objective_shape(c, self.problem.num_vars, allow_batch=True, full=True)
        self._write_cost(c, is_full=True)

    def _write_cost(self, c, is_full):
        # convert, scatter if needed, negate for MAX, and place on the device
        prob = self.problem
        if isinstance(c, torch.Tensor):
            c = (c.detach() if self._has_jax_gpu else c.detach().cpu()).to(torch.float32)
            if is_full:
                coef = c.contiguous()
            else:
                index = torch.as_tensor(prob.c_pred_index, dtype=torch.long, device=c.device)
                coef = c.new_zeros((*c.shape[:-1], prob.num_vars)).index_add_(-1, index, c)
                coef += torch.as_tensor(prob.fixed_cost, dtype=torch.float32, device=c.device)
            self.c = jnp.from_dlpack(coef)
            if self._gpu_device is not None:
                self.c = jax.device_put(self.c, self._gpu_device)
            if self.device != self.c.device:
                self._move_to_device(self.c.device)
        else:
            arr = np.asarray(c, dtype=np.float32)
            if is_full:
                coef = arr
            else:
                coef = np.broadcast_to(prob.fixed_cost, (*arr.shape[:-1], prob.num_vars)).astype(
                    np.float32
                )
                coef[..., prob.c_pred_index] += arr
            self.c = jax.device_put(jnp.asarray(coef), self.device)
        if self.modelSense == EPO.MAXIMIZE:
            self.c = -self.c

    def solve(self):
        # the full (relaxed) decision-variable solution and true objective value
        sol, obj, status = self.jitted_solve(self.c)
        _warn_if_not_optimal(status)
        full_obj = float(obj) if self.modelSense == EPO.MINIMIZE else -float(obj)
        # bare objective constants live outside the solver model
        return torch.from_dlpack(sol), full_obj + self.problem.obj_offset

    def _add_cut(self, coef, rhs):
        # add coef @ x <= rhs  ->  -coef @ x >= -rhs  to a fresh copy
        new_model = self.copy()
        row = -jnp.asarray(np.asarray(coef, np.float32)).reshape(1, -1)
        new_model._append_inequality(row, -float(rhs))
        return new_model
