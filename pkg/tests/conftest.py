#!/usr/bin/env python
"""Shared pytest fixtures, backend probes, and skip markers for PyEPO tests.

Backend availability is probed once here and exposed as ``_HAS_*`` flags plus
ready-made ``requires_*`` skip markers, so individual test modules never
re-detect solvers. Optional backends (COPT, Pyomo, MPAX, auto-sklearn) that are
absent simply skip rather than fail.

The expensive ``optDataset`` fixtures are module-scoped: each solves one tiny
optimization problem per sample at construction, so they are built once and
shared across a whole test module.
"""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

import pyepo.func as F

# ============================================================
# Backend probes (run once at import)
# ============================================================

try:
    # instantiate, not just import: a bare import succeeds without a license
    from pyepo.model.grb.shortestpath import shortestPathModel

    shortestPathModel(grid=(3, 3))
    _HAS_GUROBI = True
except Exception:
    _HAS_GUROBI = False

try:
    import pyomo.environ  # noqa: F401

    _HAS_PYOMO = True
except Exception:
    _HAS_PYOMO = False

try:
    import coptpy  # noqa: F401

    _HAS_COPT = True
except Exception:
    _HAS_COPT = False

try:
    from pyepo.model.ort.ortmodel import _HAS_ORTOOLS
except Exception:
    _HAS_ORTOOLS = False

try:
    import jax  # noqa: F401

    _HAS_JAX = True
except Exception:
    _HAS_JAX = False

try:
    import mpax  # noqa: F401

    _HAS_MPAX = _HAS_JAX
except Exception:
    _HAS_MPAX = False

try:
    from pyepo.twostage.autosklearnpred import _HAS_AUTO
except Exception:
    _HAS_AUTO = False

try:
    import clarabel  # noqa: F401

    _HAS_CLARABEL = True
except Exception:
    _HAS_CLARABEL = False

_HAS_CUDA = torch.cuda.is_available()

# MPAX runs on JAX; the jax-gpu <-> torch-cuda dlpack path needs a GPU jax device
try:
    import jax as _jax

    _HAS_JAX_GPU = any(d.platform == "gpu" for d in _jax.devices())
except Exception:
    _HAS_JAX_GPU = False


# ============================================================
# Skip markers
# ============================================================

requires_gurobi = pytest.mark.skipif(not _HAS_GUROBI, reason="Gurobi not installed")
requires_pyomo = pytest.mark.skipif(not _HAS_PYOMO, reason="Pyomo not installed")
requires_copt = pytest.mark.skipif(not _HAS_COPT, reason="COPT not installed")
requires_ortools = pytest.mark.skipif(not _HAS_ORTOOLS, reason="OR-Tools not installed")
requires_jax = pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
requires_mpax = pytest.mark.skipif(not _HAS_MPAX, reason="MPAX (jax + mpax) not installed")
requires_clarabel = pytest.mark.skipif(not _HAS_CLARABEL, reason="Clarabel not installed")
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")
requires_jax_gpu = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_JAX_GPU), reason="CUDA + jax-gpu not both available"
)


# ============================================================
# Shared constants and prediction model
# ============================================================

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


# ============================================================
# Canonical loss/op registry (shared by the func / integration / cuda layers)
# ============================================================
# name -> (kind, build, sig)
#   kind  : "solution" -> forward returns a predicted (batch, vars) tensor
#           "loss"     -> forward returns a scalar and honours `reduction`
#   build : (optmodel, dataset, reduction) -> module; reduction is ignored by
#           solution-returning ops, which take no such argument
#   sig   : forward-argument spec for call_op, e.g. "cp", "cp,c", "cp,c,w,z"
LOSS_REGISTRY = {
    "DBB": ("solution", lambda om, ds, r: F.DBB(om, processes=1, lambd=10), "cp"),
    "NID": ("solution", lambda om, ds, r: F.NID(om, processes=1), "cp"),
    "DPO": ("solution", lambda om, ds, r: F.DPO(om, processes=1, n_samples=3, sigma=1.0), "cp"),
    "DPOMul": (
        "solution",
        lambda om, ds, r: F.DPOMul(om, processes=1, n_samples=3, sigma=0.5),
        "cp",
    ),
    "IMLE": ("solution", lambda om, ds, r: F.IMLE(om, processes=1, n_samples=3, sigma=1.0), "cp"),
    "AIMLE": ("solution", lambda om, ds, r: F.AIMLE(om, processes=1, n_samples=3, sigma=1.0), "cp"),
    "RFWO": ("solution", lambda om, ds, r: F.RFWO(om, processes=1, lambd=1.0, max_iter=5), "cp"),
    "SPOPlus": ("loss", lambda om, ds, r: F.SPOPlus(om, processes=1, reduction=r), "cp,c,w,z"),
    "PG": ("loss", lambda om, ds, r: F.PG(om, processes=1, sigma=1.0, reduction=r), "cp,c"),
    "PFY": (
        "loss",
        lambda om, ds, r: F.PFY(om, processes=1, n_samples=3, sigma=1.0, reduction=r),
        "cp,w",
    ),
    "PFYMul": (
        "loss",
        lambda om, ds, r: F.PFYMul(om, processes=1, n_samples=3, sigma=0.5, reduction=r),
        "cp,w",
    ),
    "RFY": (
        "loss",
        lambda om, ds, r: F.RFY(om, processes=1, lambd=1.0, max_iter=5, reduction=r),
        "cp,w",
    ),
    "NCE": (
        "loss",
        lambda om, ds, r: F.NCE(om, processes=1, solve_ratio=1, dataset=ds, reduction=r),
        "cp,w",
    ),
    "CMAP": (
        "loss",
        lambda om, ds, r: F.CMAP(om, processes=1, solve_ratio=1, dataset=ds, reduction=r),
        "cp,w",
    ),
    "lsLTR": (
        "loss",
        lambda om, ds, r: F.lsLTR(om, processes=1, solve_ratio=1, dataset=ds, reduction=r),
        "cp,c",
    ),
    "prLTR": (
        "loss",
        lambda om, ds, r: F.prLTR(om, processes=1, solve_ratio=1, dataset=ds, reduction=r),
        "cp,c",
    ),
    "ptLTR": (
        "loss",
        lambda om, ds, r: F.ptLTR(om, processes=1, solve_ratio=1, dataset=ds, reduction=r),
        "cp,c",
    ),
}

SOLUTION_OPS = [n for n, (kind, _, _) in LOSS_REGISTRY.items() if kind == "solution"]
LOSS_OPS = [n for n, (kind, _, _) in LOSS_REGISTRY.items() if kind == "loss"]


# ============================================================
# JAX loss registry (mirrors LOSS_REGISTRY 1:1 for the pyepo.func.jax frontend)
# ============================================================
# Same acronym keys / sigs as LOSS_REGISTRY; build swaps to pyepo.func.jax. The
# shared contract tests construct from this, so a drifted jax signature fails
# to build. Guarded behind the jax import so torch-only collection is unaffected.
if _HAS_JAX:
    import pyepo.func.jax as JF

    JAX_LOSS_REGISTRY = {
        "DBB": ("solution", lambda om, ds, r: JF.DBB(om, lambd=10), "cp"),
        "NID": ("solution", lambda om, ds, r: JF.NID(om), "cp"),
        "DPO": ("solution", lambda om, ds, r: JF.DPO(om, n_samples=3, sigma=1.0), "cp"),
        "DPOMul": ("solution", lambda om, ds, r: JF.DPOMul(om, n_samples=3, sigma=0.5), "cp"),
        "IMLE": ("solution", lambda om, ds, r: JF.IMLE(om, n_samples=3, sigma=1.0), "cp"),
        "AIMLE": ("solution", lambda om, ds, r: JF.AIMLE(om, n_samples=3, sigma=1.0), "cp"),
        "RFWO": ("solution", lambda om, ds, r: JF.RFWO(om, lambd=1.0, max_iter=5), "cp"),
        "SPOPlus": ("loss", lambda om, ds, r: JF.SPOPlus(om, reduction=r), "cp,c,w,z"),
        "PG": ("loss", lambda om, ds, r: JF.PG(om, sigma=1.0, reduction=r), "cp,c"),
        "PFY": ("loss", lambda om, ds, r: JF.PFY(om, n_samples=3, sigma=1.0, reduction=r), "cp,w"),
        "PFYMul": (
            "loss",
            lambda om, ds, r: JF.PFYMul(om, n_samples=3, sigma=0.5, reduction=r),
            "cp,w",
        ),
        "RFY": ("loss", lambda om, ds, r: JF.RFY(om, lambd=1.0, max_iter=5, reduction=r), "cp,w"),
        "NCE": ("loss", lambda om, ds, r: JF.NCE(om, dataset=ds, reduction=r), "cp,w"),
        "CMAP": ("loss", lambda om, ds, r: JF.CMAP(om, dataset=ds, reduction=r), "cp,w"),
        "lsLTR": ("loss", lambda om, ds, r: JF.lsLTR(om, dataset=ds, reduction=r), "cp,c"),
        "prLTR": ("loss", lambda om, ds, r: JF.prLTR(om, dataset=ds, reduction=r), "cp,c"),
        "ptLTR": ("loss", lambda om, ds, r: JF.ptLTR(om, dataset=ds, reduction=r), "cp,c"),
    }
    JAX_SOLUTION_OPS = [n for n, (k, _, _) in JAX_LOSS_REGISTRY.items() if k == "solution"]
    JAX_LOSS_OPS = [n for n, (k, _, _) in JAX_LOSS_REGISTRY.items() if k == "loss"]
else:
    JAX_LOSS_REGISTRY = {}
    JAX_SOLUTION_OPS = []
    JAX_LOSS_OPS = []


def to_np(sol):
    """Normalize a solver solution to CPU numpy (MPAX may return a CUDA torch.Tensor)."""
    if hasattr(sol, "cpu"):
        return sol.cpu().numpy()
    return np.asarray(sol)


def solver_atol(optmodel, exact=1e-3, first_order=5e-2):
    """Gradient-gate tolerance for a backend: exact solvers reproduce to 1e-3;
    MPAX is first-order PDHG, where GPU-nondeterministic reductions can flip a
    near-tie vertex between two solves of the same costs (observed ~1.5e-2)."""
    if optmodel.__class__.__module__.startswith("pyepo.model.mpax"):
        return first_order
    return exact


def take_batch(loader, n=4):
    """First n rows of the loader's first batch.

    Returns the PyEPO sample tuple (x, c, w, z) = (features, true cost, optimal
    solution, optimal objective). Throughout the tests `cp` denotes the predicted
    cost that a loss differentiates w.r.t.
    """
    x, c, w, z = next(iter(loader))
    return x[:n], c[:n], w[:n], z[:n]


def call_op(fn, sig, cp, c, w, z):
    """Call a loss/op module with the forward args named in `sig` (see take_batch
    for the c/w/z naming; cp is the predicted cost)."""
    args = {"cp": cp, "c": c, "w": w, "z": z}
    return fn(*(args[a] for a in sig.split(",")))


def to_jax(*arrays):
    """torch tensors -> jax arrays."""
    import jax.numpy as jnp

    return tuple(jnp.asarray(a.numpy()) for a in arrays)


def sp_jax_pred(backend="mpax", n=16):
    """Shortest-path (model, dataset, pred, true_cost, true_sol, true_obj) for a JAX backend.

    pred is 1.3 x the true cost; the cost/sol/obj triple is returned as float32 numpy.
    """
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
    ds = optDataset(model, x, c)
    pred = (np.asarray(ds.costs) * 1.3).astype(np.float32)
    tc, ts, to = (np.asarray(a, np.float32) for a in (ds.costs, ds.sols, ds.objs))
    return model, ds, pred, tc, ts, to


def finite_diff_grad(loss_fn, x, eps=1e-3):
    """Central finite-difference gradient of a scalar loss at numpy `x`.

    `loss_fn` maps a numpy array of x's shape to a python float; returns the
    numerical gradient as a numpy array.
    """
    import numpy as np

    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (loss_fn(xp) - loss_fn(xm)) / (2 * eps)
        it.iternext()
    return grad


# ============================================================
# Backend adapters for the shared forward/backward contract (both frontends)
# ============================================================
# The contract assertions are identical on torch and jax; only the autodiff
# harness differs (torch .backward()/.grad vs jax.grad). The op build signature
# and call_op are shared, so the contract body is backend-agnostic.
class _ContractTorch:
    name = "torch"
    registry = LOSS_REGISTRY

    def inputs(self, c, w, z):
        cp = (c * 1.2).clone().detach().requires_grad_(True)
        return cp, c, w, z

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
        import torch

        return bool(torch.isfinite(x).all())


class _ContractJax:
    name = "jax"

    @property
    def registry(self):
        return JAX_LOSS_REGISTRY

    def inputs(self, c, w, z):
        return to_jax(c * 1.2, c, w, z)

    def forward(self, op, sig, cp, c, w, z):
        return call_op(op, sig, cp, c, w, z)

    def grad(self, op, sig, cp, c, w, z):
        import jax
        import jax.numpy as jnp

        return jax.grad(lambda p: jnp.sum(call_op(op, sig, p, c, w, z)))(cp)

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


@pytest.fixture(
    params=[
        pytest.param("torch"),
        pytest.param("jax", marks=requires_jax),
    ]
)
def contract_backend(request):
    """Adapter exposing one frontend's autodiff harness to the shared contract."""
    return _ContractTorch() if request.param == "torch" else _ContractJax()


# losses gated by a finite difference of the loss value; all are pool-based, so
# the gate sets solve_ratio = 0 (freezes the cached pool) to keep the value
# deterministic across the FD evaluations
FD_LOSSES = ["lsLTR", "prLTR", "ptLTR", "NCE", "CMAP"]


# ============================================================
# Module-scoped datasets (built once per module; solve at construction)
# ============================================================


@pytest.fixture(scope="module")
def sp_data():
    """Shortest-path optDataset + loader (CPU). Skipped if Gurobi missing."""
    if not _HAS_GUROBI:
        pytest.skip("Gurobi not installed")
    import pyepo
    from pyepo.data.dataset import optDataset
    from pyepo.model.grb.shortestpath import shortestPathModel

    x, c = pyepo.data.shortestpath.genData(NUM_DATA, NUM_FEAT, GRID, seed=42)
    optmodel = shortestPathModel(grid=GRID)
    dataset = optDataset(optmodel, x, c)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)
    return optmodel, dataset, loader


@pytest.fixture(scope="module")
def ks_data():
    """Knapsack optDataset + loader (CPU, MAXIMIZE). Skipped if Gurobi missing."""
    if not _HAS_GUROBI:
        pytest.skip("Gurobi not installed")
    import pyepo
    from pyepo.data.dataset import optDataset
    from pyepo.model.grb.knapsack import knapsackModel

    weights, x, c = pyepo.data.knapsack.genData(NUM_DATA, NUM_FEAT, 4, dim=1, deg=1, seed=42)
    optmodel = knapsackModel(weights=weights, capacity=[10.0])
    dataset = optDataset(optmodel, x, c)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)
    return optmodel, dataset, loader


# ------------------------------------------------------------
# Cross-backend pipeline fixtures
# ------------------------------------------------------------
# Gurobi is already exhaustively covered (test_30/50/60/80), so the pipeline
# targets the *other* backends: it drives a representative slice of the pipeline
# (dataset -> loss -> metric -> train) once per non-Gurobi backend, covering the
# predict->solve->loss path — including MPAX's batched dlpack bridge and the
# MAXIMIZE sign flip — that a Gurobi-only suite never reaches.
# Pyomo is deliberately absent: the legacy omo models wrap the Gurobi solver, so
# the pipeline would re-drive the same solve path the Gurobi layers already cover.

_PIPELINE_BACKENDS = [
    pytest.param("copt", marks=requires_copt),
    pytest.param("ort", marks=requires_ortools),
    pytest.param("mpax", marks=requires_mpax),
]

# backends for the gradient-truth gates: every installed backend, Gurobi included
_SP_TRUTH_BACKENDS = [pytest.param("grb", marks=requires_gurobi), *_PIPELINE_BACKENDS]


def _sp_optmodel(backend):
    """Build a shortest-path optModel (LP) on the given backend."""
    if backend == "grb":
        from pyepo.model.grb.shortestpath import shortestPathModel
    elif backend == "copt":
        from pyepo.model.copt.shortestpath import shortestPathModel
    elif backend == "ort":
        from pyepo.model.ort.shortestpath import shortestPathModel
    elif backend == "mpax":
        from pyepo.model.mpax.shortestpath import shortestPathModel
    else:
        raise ValueError(backend)
    return shortestPathModel(grid=GRID)


def _sp_dataset(backend):
    """Shortest-path optDataset + loader on the given backend."""
    import pyepo
    from pyepo.data.dataset import optDataset

    optmodel = _sp_optmodel(backend)
    x, c = pyepo.data.shortestpath.genData(NUM_DATA, NUM_FEAT, GRID, seed=42)
    dataset = optDataset(optmodel, x, c)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)
    return optmodel, dataset, loader


def _ks_optmodel(backend, weights):
    """Build a knapsack optModel (MAXIMIZE) on the given backend."""
    if backend == "grb":
        from pyepo.model.grb.knapsack import knapsackModel
    elif backend == "copt":
        from pyepo.model.copt.knapsack import knapsackModel
    elif backend == "ort":
        from pyepo.model.ort.knapsack import knapsackModel
    elif backend == "mpax":
        from pyepo.model.mpax.knapsack import knapsackModel
    else:
        raise ValueError(backend)
    return knapsackModel(weights=weights, capacity=[10.0])


@pytest.fixture(scope="module", params=_PIPELINE_BACKENDS)
def sp_pipeline(request):
    """Shortest-path optDataset + loader, one instance per installed backend."""
    return _sp_dataset(request.param)


@pytest.fixture(scope="module", params=_SP_TRUTH_BACKENDS)
def sp_truth(request):
    """Shortest-path optDataset on every installed backend (Gurobi included)."""
    return _sp_dataset(request.param)


@pytest.fixture(scope="module", params=_PIPELINE_BACKENDS)
def ks_pipeline(request):
    """Knapsack (MAXIMIZE) optDataset + loader, one instance per installed backend."""
    import pyepo
    from pyepo.data.dataset import optDataset

    weights, x, c = pyepo.data.knapsack.genData(NUM_DATA, NUM_FEAT, 4, dim=1, deg=1, seed=42)
    optmodel = _ks_optmodel(request.param, weights)
    dataset = optDataset(optmodel, x, c)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)
    return optmodel, dataset, loader


@pytest.fixture(scope="module")
def sp_constrs_data():
    """Shortest-path optDatasetConstrs + collated loader for CaVE. Gurobi only.

    Small (8 samples) because each sample triggers a Gurobi solve plus
    binding-constraint extraction.
    """
    if not _HAS_GUROBI:
        pytest.skip("Gurobi not installed")
    import pyepo
    from pyepo.data.dataset import collate_tight_constraints, optDatasetConstrs
    from pyepo.model.grb.shortestpath import shortestPathModel

    x, c = pyepo.data.shortestpath.genData(8, NUM_FEAT, GRID, seed=42)
    optmodel = shortestPathModel(grid=GRID)
    dataset = optDatasetConstrs(optmodel, x, c)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_tight_constraints)
    return optmodel, dataset, loader
