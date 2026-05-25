#!/usr/bin/env python
"""
Abstract optimization model based on MPAX
"""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING

import torch

try:
    import jax
    from jax import numpy as jnp
    from mpax import create_lp, raPDHG

    _HAS_MPAX = True
except ImportError:
    _HAS_MPAX = False

from pyepo import EPO
from pyepo.model.opt import optModel

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


class optMpaxModel(optModel):
    """
    This is an abstract class for an MPAX-based optimization model.

    Concrete subclasses populate the constraint matrices and bounds inside
    ``_getModel``:

        def _getModel(self):
            self.A = jnp.array(...)   # equality A x = b
            self.b = jnp.array(...)
            self.G = jnp.array(...)   # inequality G x >= h
            self.h = jnp.array(...)
            self.l = jnp.array(...)   # variable lower bound
            self.u = jnp.array(...)   # variable upper bound
            return None, []

    Sense (MIN / MAX) follows ``self.modelSense`` (set by a problem-level base
    such as ``knapsackBase`` before this ``__init__`` runs; defaults to MIN).
    Sparse-matrix format can be toggled by overriding the class attribute
    ``use_sparse_matrix`` on the concrete subclass.

    Attributes:
        A (jnp.ndarray): The matrix of equality constraints.
        b (jnp.ndarray): The right hand side of equality constraints.
        G (jnp.ndarray): The matrix for inequality constraints.
        h (jnp.ndarray): The right hand side of inequality constraints.
        l (jnp.ndarray): The lower bound of the variables.
        u (jnp.ndarray): The upper bound of the variables.
        use_sparse_matrix (bool): Whether to use sparse matrix format.
    """

    use_sparse_matrix: bool = True

    def __init__(self) -> None:
        if not _HAS_MPAX:
            raise ImportError("MPAX is not installed. Please install MPAX to use this feature.")
        super().__init__()  # → optModel.__init__ → self._getModel() populates A/b/G/h/l/u
        # init device
        self.device = None
        # cache GPU availability
        self._has_jax_gpu = any(d.platform == "gpu" for d in jax.devices())
        # JIT pre-compile
        self._rebuild_jit()

    def __repr__(self) -> str:
        return "optMpaxModel " + self.__class__.__name__

    def _rebuild_jit(self) -> None:
        """Rebuild JIT-compiled solve functions with current constraints."""
        self.jitted_solve = jax.jit(
            partial(
                self._jitted_solve,
                A=self.A,
                b=self.b,
                G=self.G,
                h=self.h,
                l=self.l,
                u=self.u,
                use_sparse_matrix=self.use_sparse_matrix,
            )
        )
        self.batch_optimize = jax.vmap(self.jitted_solve)

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
            # convert PyTorch tensor to JAX array using DLPack
            self.c = jnp.from_dlpack(c)
            # move constraints and bounds to device
            if self.device != self.c.device:
                self.device = self.c.device
                self.A = jax.device_put(self.A, self.device)
                self.b = jax.device_put(self.b, self.device)
                self.G = jax.device_put(self.G, self.device)
                self.h = jax.device_put(self.h, self.device)
                self.l = jax.device_put(self.l, self.device)
                self.u = jax.device_put(self.u, self.device)
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
        sol, obj = self.jitted_solve(self.c)
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
    def _jitted_solve(c, A, b, G, h, l, u, use_sparse_matrix):
        """
        A static method for JIT compile
        """
        lp = create_lp(c, A, b, G, h, l, u, use_sparse_matrix=use_sparse_matrix)
        solver = raPDHG(eps_abs=1e-4, eps_rel=1e-4, verbose=False)
        result = solver.optimize(lp)
        obj = jnp.dot(c, result.primal_solution)
        return result.primal_solution, obj

    def copy(self) -> Self:
        """
        A method to copy the model

        Returns:
            optModel: new copied model
        """
        # remove device to avoid error
        device = self.device
        self.device = None
        # copy new model
        new_model = deepcopy(self)
        # restore device
        self.device = device
        new_model.device = device
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
