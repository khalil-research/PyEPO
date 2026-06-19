"""Single declarative registry for the Torch and JAX loss frontends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pyepo.func as F
from tests.support.backends import _HAS_JAX

if _HAS_JAX:
    import pyepo.func.jax as JF
else:
    JF = None

OpKind = Literal["solution", "loss"]


@dataclass(frozen=True)
class OpSpec:
    """Frontend-neutral construction contract for one differentiable op."""

    kind: OpKind
    sig: str
    kwargs: dict = field(default_factory=dict)
    needs_dataset: bool = False


OP_SPECS = {
    "DBB": OpSpec("solution", "cp", {"lambd": 10}),
    "NID": OpSpec("solution", "cp"),
    "DPO": OpSpec("solution", "cp", {"n_samples": 3, "sigma": 1.0}),
    "DPOMul": OpSpec("solution", "cp", {"n_samples": 3, "sigma": 0.5}),
    "IMLE": OpSpec("solution", "cp", {"n_samples": 3, "sigma": 1.0}),
    "AIMLE": OpSpec("solution", "cp", {"n_samples": 3, "sigma": 1.0}),
    "RFWO": OpSpec("solution", "cp", {"lambd": 1.0, "max_iter": 5}),
    "SPOPlus": OpSpec("loss", "cp,c,w,z"),
    "PG": OpSpec("loss", "cp,c", {"sigma": 1.0}),
    "PFY": OpSpec("loss", "cp,w", {"n_samples": 3, "sigma": 1.0}),
    "PFYMul": OpSpec("loss", "cp,w", {"n_samples": 3, "sigma": 0.5}),
    "RFY": OpSpec("loss", "cp,w", {"lambd": 1.0, "max_iter": 5}),
    "NCE": OpSpec("loss", "cp,w", needs_dataset=True),
    "CMAP": OpSpec("loss", "cp,w", needs_dataset=True),
    "lsLTR": OpSpec("loss", "cp,c", needs_dataset=True),
    "prLTR": OpSpec("loss", "cp,c", needs_dataset=True),
    "ptLTR": OpSpec("loss", "cp,c", needs_dataset=True),
}


def _build(frontend, name, optmodel, dataset, reduction):
    spec = OP_SPECS[name]
    kwargs = dict(spec.kwargs)
    if frontend is F:
        kwargs["processes"] = 1
    if spec.kind == "loss":
        kwargs["reduction"] = reduction
    if spec.needs_dataset:
        kwargs["dataset"] = dataset
        kwargs["solve_ratio"] = 1
    return getattr(frontend, name)(optmodel, **kwargs)


def _registry(frontend):
    if frontend is None:
        return {}
    return {
        name: (
            spec.kind,
            lambda om, ds, reduction, name=name: _build(frontend, name, om, ds, reduction),
            spec.sig,
        )
        for name, spec in OP_SPECS.items()
    }


LOSS_REGISTRY = _registry(F)
JAX_LOSS_REGISTRY = _registry(JF)
SOLUTION_OPS = [name for name, spec in OP_SPECS.items() if spec.kind == "solution"]
LOSS_OPS = [name for name, spec in OP_SPECS.items() if spec.kind == "loss"]
JAX_SOLUTION_OPS = SOLUTION_OPS if _HAS_JAX else []
JAX_LOSS_OPS = LOSS_OPS if _HAS_JAX else []
FD_LOSSES = ["lsLTR", "prLTR", "ptLTR", "NCE", "CMAP"]

__all__ = [
    "FD_LOSSES",
    "JAX_LOSS_OPS",
    "JAX_LOSS_REGISTRY",
    "JAX_SOLUTION_OPS",
    "LOSS_OPS",
    "LOSS_REGISTRY",
    "OP_SPECS",
    "SOLUTION_OPS",
]
