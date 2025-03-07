#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on MPAX
"""

from copy import deepcopy
from functools import partial
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


class optMpaxModel(optModel):
    """
    This is an abstract class for MPAX-based optimization model

    Attributes:
        A (jnp.ndarray, BCOO or BCSR): The matrix of equality constraints.
        b (jnp.ndarray): The right hand side of equality constraints.
        G (jnp.ndarray, BCOO or BCSR): The matrix for inequality constraints.
        h (jnp.ndarray): The right hand side of inequality constraints.
        l (jnp.ndarray): The lower bound of the variables.
        u (jnp.ndarray): The upper bound of the variables.
        use_sparse_matrix (bool): Whether to use sparse matrix format, by default True.
        minimize (bool): Whether to minimize objective, by default True.
    """

    def __init__(self, A=None, b=None, G=None, h=None, l=None, u=None, use_sparse_matrix=True, minimize=True):
        super().__init__()
        # error
        if not _HAS_MPAX:
            raise ImportError("MPAX is not installed. Please install MPAX to use this feature.")
         # at least one of A or G must be provided
        if A is None and G is None:
            raise ValueError("At least one of A (equality constraints) or G (inequality constraints) must be provided.")
        # rhs is provided
        if A is not None and b is None:
            raise ValueError("If A (equality constraints) is provided, b must also be provided.")
        if G is not None and h is None:
            raise ValueError("If G (inequality constraints) is provided, h must also be provided.")
        # number of variables
        num_vars = A.shape[1] if A is not None else G.shape[1]
        # params for equality constraints (A x = b)
        if A is not None and b is not None:
            self.A = jnp.array(A, dtype=jnp.float32)
            self.b = jnp.array(b, dtype=jnp.float32)
        else:
            # no equality constraints: use empty arrays
            self.A = jnp.zeros((0, num_vars), dtype=jnp.float32)
            self.b = jnp.zeros((0,), dtype=jnp.float32)
        # params for inequality constraints (G x >= h)
        if G is not None and h is not None:
            self.G = jnp.array(G, dtype=jnp.float32)
            self.h = jnp.array(h, dtype=jnp.float32)
        else:
            # no inequality constraints: use empty arrays
            self.G = jnp.zeros((0, num_vars), dtype=jnp.float32)
            self.h = jnp.zeros((0,), dtype=jnp.float32)
        # variable bounds
        self.l = jnp.array(l, dtype=jnp.float32) if l is not None else jnp.zeros(num_vars, dtype=jnp.float32)
        self.u = jnp.array(u, dtype=jnp.float32) if u is not None else jnp.full(num_vars, jnp.inf, dtype=jnp.float32)
        # matrix type
        self.use_sparse_matrix = use_sparse_matrix
        # model sense
        self.modelSense = EPO.MINIMIZE if minimize else EPO.MAXIMIZE
        # init device
        self.device = None
        # jit pre complile
        self.jitted_solve = jax.jit(partial(self._jitted_solve,
                                            A=self.A, b=self.b, G=self.G,
                                            h=self.h, l=self.l, u=self.u,
                                            use_sparse_matrix=self.use_sparse_matrix))
        self.batch_optimize = jax.vmap(self.jitted_solve)

    def __repr__(self):
        return "optMpaxModel " + self.__class__.__name__

    @property
    def num_cost(self):
        """
        number of cost to be predicted
        """
        return self.A.shape[1]

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (np.ndarray / list): cost of objective function
        """
        # check if c is a PyTorch tensor
        if isinstance(c, torch.Tensor):
            device = c.device  # get tensor device
            # check JAX supports CUDA
            if not any(device.platform == "gpu" for device in jax.devices()):
                # move to cpu
                c = c.cpu().detach()
            else:
                # stay in gpu
                c = c.detach()
            # convert PyTorch tensor to JAX array using DLPack
            self.c = jax.dlpack.from_dlpack(c)
            # move constraints and bounds to device
            if self.device != self.c.device:
                self.device = self.c.device
                self.A = jax.device_put(self.A, self.device)
                self.b = jax.device_put(self.b, self.device)
                self.G = jax.device_put(self.G, self.device)
                self.h = jax.device_put(self.h, self.device)
                self.l = jax.device_put(self.l, self.device)
                self.u = jax.device_put(self.u, self.device)
                # jit pre complile
                self.jitted_solve = jax.jit(partial(self._jitted_solve,
                                                    A=self.A, b=self.b, G=self.G,
                                                    h=self.h, l=self.l, u=self.u,
                                                    use_sparse_matrix=self.use_sparse_matrix))
                self.batch_optimize = jax.vmap(self.jitted_solve)
        # c is already a NumPy array
        else:
            self.c = jnp.array(c, dtype=jnp.float32)
        if c.shape[-1] != self.num_cost:
            raise ValueError("Size of cost vector cannot match vars.")
        # change sign for model sense
        if self.modelSense == EPO.MINIMIZE:
            self.c = self.c
        elif self.modelSense == EPO.MAXIMIZE:
            self.c = - self.c
        else:
            raise ValueError("Invalid modelSense.")

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (jnp.float32)
        """
        # create lp model
        sol, obj = self.jitted_solve(self.c)
        # convert to torch
        sol = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(sol))
        if self.modelSense == EPO.MINIMIZE:
            obj = obj.item()
        elif self.modelSense == EPO.MAXIMIZE:
            obj = - obj.item()
        else:
            raise ValueError("Invalid modelSense.")
        return sol, obj

    @staticmethod
    def _jitted_solve(c, A, b, G, h, l, u, use_sparse_matrix):
        """
        A static method for JIT complile
        """
        lp = create_lp(c, A, b, G, h, l, u, use_sparse_matrix=use_sparse_matrix)
        solver = raPDHG(eps_abs=1e-4, eps_rel=1e-4, verbose=False)
        result = solver.optimize(lp)
        obj = jnp.dot(c, result.primal_solution)
        return result.primal_solution, obj

    def copy(self):
        """
        A method to copy model

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

    def addConstr(self, coefs, rhs):
        """
        A method to add new constraint

        Args:
            coefs (np.ndarray / list): coeffcients of new constraint
            rhs (jnp.float32): right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        if len(coefs) != self.num_cost:
            raise ValueError("Size of coef vector cannot cost.")
        # convert coefs to jnp
        coefs = jnp.array(coefs, dtype=jnp.float32).reshape(1, -1)
        # copy
        new_model = self.copy()
        # add constraint
        if new_model.G.shape[0] == 0:
            new_model.G = coefs.reshape(1, -1)
            new_model.h = jnp.array([rhs])
        else:
            new_model.G = jnp.vstack([new_model.G, coefs])
            new_model.h = jnp.append(new_model.h, rhs)
        return new_model

    def _getModel(self):
        """
        Placeholder method for MPAX. MPAX does not require an explicit model creation.
        """
        return None, None


if __name__ == "__main__":
    import random
    import numpy as np
    import torch
    # random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    # number of variables
    num_vars = 10
    # random equality and inequality constraints
    A = np.random.rand(3, num_vars)  # 3 equality constraints
    b = np.random.rand(3)
    G = np.random.rand(5, num_vars)  # 5 inequality constraints
    h = np.random.rand(5)
    #l = np.zeros(num_vars)  # Lower bounds (default zero)
    #u = np.ones(num_vars) * 10  # Upper bounds
    # create optimization model
    optmodel = optMpaxModel(A=A, b=b, G=G, h=h, minimize=True)
    # generate a random cost vector
    cost = np.random.rand(num_vars)
    # solve the model
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    # print results
    print(f"Objective Value: {obj}")
    print(f"Solution: {sol}")

    # cpu tensor
    cost_cpu = torch.tensor(cost, dtype=torch.float32, device="cpu")
    # solve the model
    optmodel.setObj(cost_cpu)
    sol, obj = optmodel.solve()
    # print results
    print(f"Objective Value: {obj}")
    print(f"Solution: {sol}")

    # gpu tensor
    cost_gpu = torch.tensor(cost, dtype=torch.float32, device="cuda")
    # solve the model
    optmodel.setObj(cost_gpu)
    sol, obj = optmodel.solve()
    # print results
    print(f"Objective Value: {obj}")
    print(f"Solution: {sol}")

    # add a new constraint (sum of variables should be ≤ num_vars * 0.5)
    new_constraint = [1] * num_vars  # All coefficients are 1
    new_rhs = num_vars * 0.5  # Right-hand side
    optmodel = optmodel.addConstr(new_constraint, new_rhs)
    # solve again
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    # Print updated results
    print(f"Objective Value after adding constraint: {obj}")
    print(f"Updated Solution: {sol}")
