"""Shared constants, numerical helpers, and frontend contract adapters."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tests.support.registry import JAX_LOSS_REGISTRY, LOSS_REGISTRY

NUM_DATA = 32
NUM_FEAT = 3
GRID = (3, 3)
BATCH = 16


class LinearPred(nn.Module):
    """Minimal linear predictor: features -> per-cost-coefficient vector."""

    def __init__(self, num_feat, num_cost):
        super().__init__()
        self.linear = nn.Linear(num_feat, num_cost)

    def forward(self, x):
        return self.linear(x)


def to_np(sol):
    """Normalize a solver solution to CPU numpy."""
    if hasattr(sol, "cpu"):
        return sol.cpu().numpy()
    return np.asarray(sol)


def solver_atol(optmodel, exact=1e-3, first_order=5e-2):
    """Tolerance for exact versus first-order solver gradient gates."""
    if optmodel.__class__.__module__.startswith("pyepo.model.mpax"):
        return first_order
    return exact


def take_batch(loader, n=4):
    """Return the first ``n`` rows of the loader's first batch."""
    x, c, w, z = next(iter(loader))
    return x[:n], c[:n], w[:n], z[:n]


def max_knapsack():
    """MAXIMIZE knapsack shared by the torch and jax truth gates."""
    from pyepo.model.grb.knapsack import knapsackModel

    return knapsackModel(weights=[[3.0, 4.0, 2.0, 5.0, 3.0]], capacity=[10.0])


def fw_knapsack():
    """Small MAXIMIZE knapsack shared by the Frank-Wolfe gates."""
    from pyepo.model.grb.knapsack import knapsackModel

    return knapsackModel(weights=[[3.0, 4.0, 2.0, 5.0]], capacity=[7.0])


def partial_dsl_model():
    """Partial-prediction DSL model (3 predicted items + 2 fixed-cost extras)."""
    from pyepo import EPO, dsl

    items = dsl.Variable(3, vtype=EPO.BINARY)
    extra = dsl.Variable(2, vtype=EPO.BINARY)
    cost = dsl.Parameter(3)
    prob = dsl.Problem(
        dsl.Maximize(cost @ items + np.array([1.0, 2.0]) @ extra),
        [items.sum() + extra.sum() <= 3],
    )
    return prob.compile(backend="gurobi")


def call_op(fn, sig, cp, c, w, z):
    """Call an op using its compact forward-argument signature."""
    args = {"cp": cp, "c": c, "w": w, "z": z}
    return fn(*(args[arg] for arg in sig.split(",")))


def to_jax(*arrays):
    """Convert torch tensors to JAX arrays."""
    import jax.numpy as jnp

    return tuple(jnp.asarray(array.numpy()) for array in arrays)


def sp_jax_pred(backend="mpax", n=16):
    """Build shortest-path data for a JAX frontend test."""
    import pyepo
    from pyepo.data.dataset import optDataset

    if backend == "mpax":
        from pyepo.model.mpax.shortestpath import shortestPathModel
    elif backend == "grb":
        from pyepo.model.grb.shortestpath import shortestPathModel
    else:
        raise ValueError(backend)
    x, c = pyepo.data.shortestpath.genData(n, NUM_FEAT, GRID, seed=42)
    model = shortestPathModel(grid=GRID)
    dataset = optDataset(model, x, c)
    pred = (np.asarray(dataset.costs) * 1.3).astype(np.float32)
    true = tuple(
        np.asarray(array, np.float32) for array in (dataset.costs, dataset.sols, dataset.objs)
    )
    return model, dataset, pred, *true


def finite_diff_grad(loss_fn, x, eps=1e-3):
    """Central finite-difference gradient of a scalar loss."""
    grad = np.zeros_like(x)
    iterator = np.nditer(x, flags=["multi_index"])
    while not iterator.finished:
        index = iterator.multi_index
        xp = x.copy()
        xm = x.copy()
        xp[index] += eps
        xm[index] -= eps
        grad[index] = (loss_fn(xp) - loss_fn(xm)) / (2 * eps)
        iterator.iternext()
    return grad


class _ContractTorch:
    """Torch autodiff harness for the shared contract."""

    name = "torch"
    registry = LOSS_REGISTRY

    def inputs(self, c, w, z):
        return (c * 1.2).clone().detach().requires_grad_(True), c, w, z

    def forward(self, op, sig, cp, c, w, z):
        return call_op(op, sig, cp, c, w, z)

    def grad(self, op, sig, cp, c, w, z):
        out = call_op(op, sig, cp, c, w, z)
        (out if out.dim() == 0 else out.sum()).backward()
        return cp.grad

    @staticmethod
    def shape(x):
        return tuple(x.shape)

    @staticmethod
    def ndim(x):
        return x.dim()

    @staticmethod
    def to_np(x):
        return x.detach().numpy()

    @staticmethod
    def finite(x):
        return bool(torch.isfinite(x).all())


class _ContractJax:
    """JAX autodiff harness for the shared contract."""

    name = "jax"
    registry = JAX_LOSS_REGISTRY

    def inputs(self, c, w, z):
        return to_jax(c * 1.2, c, w, z)

    def forward(self, op, sig, cp, c, w, z):
        return call_op(op, sig, cp, c, w, z)

    def grad(self, op, sig, cp, c, w, z):
        import jax
        import jax.numpy as jnp

        return jax.grad(lambda pred: jnp.sum(call_op(op, sig, pred, c, w, z)))(cp)

    @staticmethod
    def shape(x):
        return tuple(x.shape)

    @staticmethod
    def ndim(x):
        return x.ndim

    @staticmethod
    def to_np(x):
        return np.asarray(x)

    @staticmethod
    def finite(x):
        return bool(np.isfinite(np.asarray(x)).all())


__all__ = [
    "BATCH",
    "GRID",
    "NUM_DATA",
    "NUM_FEAT",
    "LinearPred",
    "call_op",
    "finite_diff_grad",
    "fw_knapsack",
    "max_knapsack",
    "partial_dsl_model",
    "solver_atol",
    "sp_jax_pred",
    "take_batch",
    "to_np",
]
