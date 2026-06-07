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

import numpy as np
import torch

try:
    import jax
    from jax import numpy as jnp
except ImportError:
    pass

from pyepo import EPO
from pyepo.dsl.compiled import compiledBase
from pyepo.model.mpax.mpaxmodel import optMpaxModel


def compileProblem(problem, **params) -> compiledMpaxProblem:
    """Instantiate the MPAX-compiled problem."""
    return compiledMpaxProblem(problem, params=params)


class compiledMpaxProblem(compiledBase, optMpaxModel):
    """
    MPAX-backed (JAX LP / QP) compiled DSL problem.
    """

    use_sparse_matrix = False

    def _getModel(self) -> tuple:
        # assemble MPAX standard-form matrices from the finalized IR
        prob = self.problem
        self.modelSense = prob.objective.modelSense
        self._emit_constraints()
        self.l = jnp.asarray(np.where(np.isneginf(prob.var_lb), -np.inf, prob.var_lb).astype(np.float32))
        self.u = jnp.asarray(np.where(np.isposinf(prob.var_ub), np.inf, prob.var_ub).astype(np.float32))
        # quadratic objective (None ⇒ LP); Q = 2·obj_Q for MPAX's ½xᵀQx convention
        self.Q = jnp.asarray(2.0 * np.asarray(prob.obj_Q.todense(), np.float32)) if prob.obj_Q is not None else None
        return None, []

    def _emit_constraints(self):
        # split the IR constraints into equality (A x = b) and inequality (G x ≥ h) blocks
        n = self.problem.num_vars
        A_eq, b_eq, G, h = [], [], [], []
        for Q, A, sense, b in self.problem.constrs:
            if Q is not None:
                raise NotImplementedError("MPAX supports a quadratic objective only, not quadratic constraints.")
            A = np.asarray(A.todense(), dtype=np.float32)
            b = np.asarray(b, dtype=np.float32).reshape(-1)
            if sense == "==":
                A_eq.append(A)
                b_eq.append(b)
            elif sense == "<=":
                G.append(-A)                                # A x <= b  ->  -A x >= -b
                h.append(-b)
            else:
                G.append(A)                                 # A x >= b
                h.append(b)
        self.A = jnp.asarray(np.vstack(A_eq) if A_eq else np.zeros((0, n), np.float32))
        self.b = jnp.asarray(np.concatenate(b_eq) if b_eq else np.zeros(0, np.float32))
        self.G = jnp.asarray(np.vstack(G) if G else np.zeros((0, n), np.float32))
        self.h = jnp.asarray(np.concatenate(h) if h else np.zeros(0, np.float32))

    def _apply_params(self):
        # MPAX takes no DSL-level solver params
        if self.params:
            raise ValueError("MPAX backend does not accept solver params.")

    def setObj(self, c):
        # full coef = fixed_c + predicted c scattered at c_index, kept on device
        prob = self.problem
        if isinstance(c, torch.Tensor):
            c = (c.detach() if self._has_jax_gpu else c.detach().cpu()).to(torch.float32)
            full = torch.as_tensor(prob.fixed_c, dtype=torch.float32, device=c.device).clone()
            full.index_add_(0, torch.as_tensor(prob.c_index, dtype=torch.long, device=c.device), c)
            self.c = jnp.from_dlpack(full)
            if self._gpu_device is not None:
                self.c = jax.device_put(self.c, self._gpu_device)
            if self.device != self.c.device:
                self._move_to_device(self.c.device)
        else:
            full = prob.fixed_c.astype(np.float32)
            np.add.at(full, prob.c_index, np.asarray(c, dtype=np.float32))
            self.c = jnp.asarray(full)
        if self.modelSense == EPO.MAXIMIZE:
            self.c = -self.c

    def solve(self):
        # solve, project the solution onto the predicted positions, drop the fixed-cost part
        sol, obj = self.jitted_solve(self.c)
        sol_np = np.asarray(sol)
        full_obj = float(obj) if self.modelSense == EPO.MINIMIZE else -float(obj)
        w = torch.from_dlpack(sol)[self.problem.c_index]
        return w, full_obj - float(self.problem.fixed_c @ sol_np)

    def _add_cut(self, coef, rhs):
        # add coef @ x <= rhs  ->  -coef @ x >= -rhs  to a fresh copy
        new_model = self.copy()
        row = -jnp.asarray(np.asarray(coef, np.float32)).reshape(1, -1)
        new_model.G = row if new_model.G.shape[0] == 0 else jnp.vstack([new_model.G, row])
        new_model.h = jnp.append(new_model.h, -float(rhs))
        new_model._rebuild_jit()
        return new_model

    def _move_to_device(self, device):
        # move the constraint matrices / bounds onto the cost's device, rebuild the jit
        self.device = device
        self.A = jax.device_put(self.A, device)
        self.b = jax.device_put(self.b, device)
        self.G = jax.device_put(self.G, device)
        self.h = jax.device_put(self.h, device)
        self.l = jax.device_put(self.l, device)
        self.u = jax.device_put(self.u, device)
        if self.Q is not None:
            self.Q = jax.device_put(self.Q, device)
        self._rebuild_jit()
