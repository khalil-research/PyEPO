"""Shared loss registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, NamedTuple

import pyepo.func as F
from tests.support.backends import _HAS_JAX

if _HAS_JAX:
    import pyepo.func.jax as JF
else:
    JF = None

OpKind = Literal["solution", "loss"]
PartialPredictionCheck = Literal["parity", "smoke"]


@dataclass(frozen=True)
class OpSpec:
    """Frontend-neutral op spec."""

    kind: OpKind
    sig: str
    kwargs: dict = field(default_factory=dict)
    needs_dataset: bool = False
    finite_diff_truth: bool = False
    partial_prediction: PartialPredictionCheck | None = None


class RegistryEntry(NamedTuple):
    """Concrete frontend builder produced from an ``OpSpec``."""

    kind: OpKind
    build: Callable[..., object]
    sig: str


OP_SPECS = {
    "DBB": OpSpec("solution", "cp", {"lambd": 10}, partial_prediction="parity"),
    "NID": OpSpec("solution", "cp", partial_prediction="parity"),
    "DPO": OpSpec(
        "solution",
        "cp",
        {"n_samples": 3, "sigma": 1.0},
        partial_prediction="smoke",
    ),
    "DPOMul": OpSpec(
        "solution",
        "cp",
        {"n_samples": 3, "sigma": 0.5},
        partial_prediction="smoke",
    ),
    "IMLE": OpSpec(
        "solution",
        "cp",
        {"n_samples": 3, "sigma": 1.0},
        partial_prediction="smoke",
    ),
    "AIMLE": OpSpec(
        "solution",
        "cp",
        {"n_samples": 3, "sigma": 1.0},
        partial_prediction="smoke",
    ),
    "RFWO": OpSpec(
        "solution",
        "cp",
        {"lambd": 1.0, "max_iter": 5},
        partial_prediction="parity",
    ),
    "SPOPlus": OpSpec("loss", "cp,c,w,z", partial_prediction="parity"),
    "PG": OpSpec("loss", "cp,c", {"sigma": 1.0}, partial_prediction="parity"),
    "PFY": OpSpec(
        "loss",
        "cp,w",
        {"n_samples": 3, "sigma": 1.0},
        partial_prediction="smoke",
    ),
    "PFYMul": OpSpec(
        "loss",
        "cp,w",
        {"n_samples": 3, "sigma": 0.5},
        partial_prediction="smoke",
    ),
    "RFY": OpSpec(
        "loss",
        "cp,w",
        {"lambd": 1.0, "max_iter": 5},
        partial_prediction="parity",
    ),
    "NCE": OpSpec("loss", "cp,w", needs_dataset=True, finite_diff_truth=True),
    "CMAP": OpSpec("loss", "cp,w", needs_dataset=True, finite_diff_truth=True),
    "lsLTR": OpSpec("loss", "cp,c", needs_dataset=True, finite_diff_truth=True),
    "prLTR": OpSpec("loss", "cp,c", needs_dataset=True, finite_diff_truth=True),
    "ptLTR": OpSpec("loss", "cp,c", needs_dataset=True, finite_diff_truth=True),
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
        name: RegistryEntry(
            kind=spec.kind,
            build=lambda om, ds, reduction, name=name: _build(frontend, name, om, ds, reduction),
            sig=spec.sig,
        )
        for name, spec in OP_SPECS.items()
    }


LOSS_REGISTRY = _registry(F)
JAX_LOSS_REGISTRY = _registry(JF)
SOLUTION_OPS = [name for name, spec in OP_SPECS.items() if spec.kind == "solution"]
LOSS_OPS = [name for name, spec in OP_SPECS.items() if spec.kind == "loss"]
FD_LOSSES = [name for name, spec in OP_SPECS.items() if spec.finite_diff_truth]
PARTIAL_PREDICTION_PARITY_OPS = [
    name for name, spec in OP_SPECS.items() if spec.partial_prediction == "parity"
]
PARTIAL_PREDICTION_SMOKE_OPS = [
    name for name, spec in OP_SPECS.items() if spec.partial_prediction == "smoke"
]

__all__ = [
    "FD_LOSSES",
    "JAX_LOSS_REGISTRY",
    "LOSS_OPS",
    "LOSS_REGISTRY",
    "OP_SPECS",
    "PARTIAL_PREDICTION_PARITY_OPS",
    "PARTIAL_PREDICTION_SMOKE_OPS",
    "SOLUTION_OPS",
    "RegistryEntry",
]
