#!/usr/bin/env python
"""
Backend-dispatching factories for the built-in problems.

Each factory builds the requested problem on the chosen solver backend, selected
by the ``backend`` keyword (default Gurobi) rather than by importing a
backend-specific class.
"""

from __future__ import annotations

import importlib
from typing import Any

_BACKENDS: dict[str, str] = {
    "gurobi": "pyepo.model.grb",
    "copt": "pyepo.model.copt",
    "pyomo": "pyepo.model.omo",
    "ortools": "pyepo.model.ort",
    "mpax": "pyepo.model.mpax",
}

_TSP: dict[str, str] = {"DFJ": "tspDFJModel", "GG": "tspGGModel", "MTZ": "tspMTZModel"}
_VRP: dict[str, str] = {"RCI": "vrpRCIModel", "MTZ": "vrpMTZModel"}


def _class_for(backend: str, name: str) -> type[Any]:
    """Resolve a problem class by name on the selected backend."""
    try:
        module = _BACKENDS[backend]
    except KeyError:
        raise ValueError(f"unknown backend {backend!r}; choose from {list(_BACKENDS)}") from None
    cls = getattr(importlib.import_module(module), name, None)
    if cls is None:
        raise ValueError(f"this problem is not available on backend {backend!r}")
    return cls


def _formulation(table: dict[str, str], name: str, kind: str) -> str:
    """Resolve a formulation name to its backend class name."""
    try:
        return table[name]
    except KeyError:
        raise ValueError(
            f"unknown {kind} formulation {name!r}; choose from {list(table)}"
        ) from None


def shortestPathModel(grid, *, backend="gurobi", **kwargs):
    """
    Shortest path on a grid network.

    Args:
        grid (tuple): grid size ``(h, w)``
        backend (str): solver backend; one of ``"gurobi"``, ``"copt"``, ``"pyomo"``, ``"ortools"``, ``"mpax"``
    """
    return _class_for(backend, "shortestPathModel")(grid, **kwargs)


def knapsackModel(weights, capacity, *, backend="gurobi", **kwargs):
    """
    Multi-dimensional knapsack.

    Args:
        weights (ndarray): item weights with shape ``(dim, n_items)``
        capacity (ndarray): per-dimension capacity with length ``dim``
        backend (str): solver backend; one of ``"gurobi"``, ``"copt"``, ``"pyomo"``, ``"ortools"``, ``"mpax"``
    """
    return _class_for(backend, "knapsackModel")(weights, capacity, **kwargs)


def portfolioModel(num_assets, covariance, *, backend="gurobi", **kwargs):
    """
    Mean-variance portfolio optimization.

    Args:
        num_assets (int): number of assets
        covariance (ndarray): covariance matrix of the asset returns
        backend (str): solver backend; one of ``"gurobi"``, ``"copt"``, ``"pyomo"``
    """
    return _class_for(backend, "portfolioModel")(num_assets, covariance, **kwargs)


def tspModel(num_nodes, *, backend="gurobi", formulation="DFJ", **kwargs):
    """
    Traveling salesperson.

    Args:
        num_nodes (int): number of nodes
        backend (str): solver backend; one of ``"gurobi"``, ``"copt"``, ``"pyomo"``
        formulation (str): ILP formulation; one of ``"DFJ"``, ``"GG"``, ``"MTZ"`` (``"DFJ"`` on gurobi and copt only)
    """
    return _class_for(backend, _formulation(_TSP, formulation, "TSP"))(num_nodes, **kwargs)


def vrpModel(
    num_nodes, demands, capacity, num_vehicle, *, backend="gurobi", formulation="RCI", **kwargs
):
    """
    Capacitated vehicle routing.

    Args:
        num_nodes (int): number of nodes, with the depot as node 0
        demands (list): per-customer demands with length ``num_nodes - 1``
        capacity (float): vehicle capacity
        num_vehicle (int): number of vehicles
        backend (str): solver backend; one of ``"gurobi"``, ``"copt"``, ``"pyomo"``
        formulation (str): ILP formulation; ``"RCI"`` or ``"MTZ"`` (``"RCI"`` on gurobi and copt only)
    """
    cls = _class_for(backend, _formulation(_VRP, formulation, "VRP"))
    return cls(num_nodes, demands, capacity, num_vehicle, **kwargs)
