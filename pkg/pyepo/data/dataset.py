#!/usr/bin/env python
"""
optDataset class based on PyTorch Dataset
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from scipy.spatial import distance
from torch.utils.data import Dataset
from tqdm import tqdm

from pyepo import EPO
from pyepo.data._validation import validate_positive_int, validate_probability
from pyepo.model.opt import optModel

if TYPE_CHECKING:
    from pyepo.model.mpax import optMpaxModel as _optMpaxModelT

try:
    from pyepo.model.mpax import optMpaxModel
except ImportError:
    optMpaxModel = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _validate_inputs(
    model: optModel,
    feats: np.ndarray | torch.Tensor,
    costs: np.ndarray | torch.Tensor,
) -> None:
    """Validate the common constructor contract for optimization datasets."""
    if not isinstance(model, optModel):
        raise TypeError("arg model is not an optModel")
    if len(feats) != len(costs):
        raise ValueError(
            f"feats and costs must have the same number of instances: {len(feats)} vs {len(costs)}."
        )


def _as_float_tensor(data) -> torch.Tensor:
    """Convert dataset arrays to the common float32 tensor representation."""
    return torch.as_tensor(data, dtype=torch.float32)


def _solution_to_numpy(
    solution: np.ndarray | torch.Tensor | list,
) -> np.ndarray:
    """Normalize a solver solution to a NumPy array."""
    if isinstance(solution, torch.Tensor):
        solution = solution.detach().cpu().numpy()
    return np.asarray(solution)


class optDataset(Dataset):
    """
    PyTorch ``Dataset`` for predict-then-optimize problems.

    At construction time it solves the optimization problem for every cost
    vector and caches the optimal solution :math:`\\mathbf{w}^*(\\mathbf{c})`
    and objective value :math:`z^*(\\mathbf{c})`. This precomputation removes
    solver overhead from the training loop, making ``optDataset`` the standard
    input format for end-to-end training in PyEPO. When labels are already
    available from another source, ``optDataset`` can be skipped and batches
    fed directly to ``pyepo.func`` modules.

    Attributes:
        model (optModel): Optimization model
        feats (torch.Tensor): Data features
        costs (torch.Tensor): Cost vectors
        sols (torch.Tensor): Cached optimal solutions w*(c)
        objs (torch.Tensor): Cached optimal objective values z*(c)
    """

    def __init__(
        self,
        model: optModel,
        feats: np.ndarray | torch.Tensor,
        costs: np.ndarray | torch.Tensor,
    ) -> None:
        """
        Build the dataset and precompute optimal labels.

        Args:
            model: an instance of optModel
            feats: data features
            costs: costs of objective function
        """
        _validate_inputs(model, feats, costs)
        self.model = model
        # data
        self.feats = feats
        self.costs = costs
        # find optimal solutions
        sols, objs = self._get_sols()
        self.feats = _as_float_tensor(feats)
        self.costs = _as_float_tensor(costs)
        self.sols = _as_float_tensor(sols)
        self.objs = _as_float_tensor(objs)

    def _get_sols(self) -> tuple[np.ndarray, np.ndarray]:
        """
        A method to get optimal solutions for all cost vectors
        """
        # MPAX fast path: vmap-solve the whole dataset in a single dispatch
        if optMpaxModel is not None and isinstance(self.model, optMpaxModel):
            return self._get_sols_mpax_batch()
        sols = []
        objs = []
        logger.info("Optimizing for optDataset...")
        for c in tqdm(self.costs):
            sol, obj = self._solve(c)
            sols.append(_solution_to_numpy(sol))
            objs.append(obj)
        return np.stack(sols), np.asarray(objs).reshape(-1, 1)

    def _get_sols_mpax_batch(self) -> tuple[np.ndarray, np.ndarray]:
        """
        A method to batch-solve every cost vector in one MPAX vmap call.
        """
        logger.info("Optimizing for optDataset (MPAX batched)...")
        from pyepo.model.mpax.mpaxmodel import _warn_if_not_optimal

        model = cast("_optMpaxModelT", self.model)
        model._setFullObj(model._fullCost(self.costs))
        sols, objs, status = model.batch_optimize(model.c)
        _warn_if_not_optimal(status)
        # writable copy; torch.as_tensor warns on JAX read-only buffers
        sols_np = np.array(sols, dtype=np.float32)
        objs_np = np.array(objs, dtype=np.float32)
        # jitted_solve returns c·sol where the objective write already negated c for MAX
        if self.model.modelSense == EPO.MAXIMIZE:
            objs_np = -objs_np
        # compiled DSL problems carry bare objective constants outside the solver model
        problem = getattr(self.model, "problem", None)
        if problem is not None:
            objs_np += problem.obj_offset
        return sols_np, objs_np.reshape(-1, 1)

    def _solve(
        self,
        cost: np.ndarray | torch.Tensor | list,
    ) -> tuple[np.ndarray | torch.Tensor | list, float]:
        """
        A method to solve optimization problem to get an optimal solution with given cost

        Args:
            cost: cost of objective function

        Returns:
            tuple: optimal solution (np.ndarray) and objective value (float)
        """
        self.model._setFullObj(self.model._fullCost(cost))
        sol, obj = self.model.solve()
        return sol, obj

    def __len__(self) -> int:
        """
        A method to get data size

        Returns:
            int: the number of optimization problems
        """
        return len(self.costs)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        A method to retrieve data

        Args:
            index: data index

        Returns:
            tuple: data features (torch.tensor), costs (torch.tensor), optimal solutions (torch.tensor) and objective values (torch.tensor)
        """
        return (
            cast("torch.Tensor", self.feats[index]),
            cast("torch.Tensor", self.costs[index]),
            self.sols[index],
            self.objs[index],
        )


class optDatasetKNN(optDataset):
    """
    PyTorch ``Dataset`` for the kNN-robust decision-focused loss.

    For each instance the cost vector is replaced with a convex combination of
    its k nearest neighbours in feature space, and the optimization problem is
    solved on the smoothed costs. The mean kNN solutions and objective values
    are cached for training, providing a robust supervision signal under noisy
    or out-of-distribution feature observations.

    Reference: Schutte et al. (2023) `<https://arxiv.org/abs/2310.04328>`_

    Attributes:
        model (optModel): Optimization model
        k (int): number of nearest neighbours selected
        weight (float): self-weight in the kNN convex combination (1.0 = no smoothing)
        feats (torch.Tensor): Data features
        costs (torch.Tensor): kNN-smoothed cost vectors
        sols (torch.Tensor): Mean kNN optimal solutions
        objs (torch.Tensor): Mean kNN optimal objective values
    """

    def __init__(
        self,
        model: optModel,
        feats: np.ndarray | torch.Tensor,
        costs: np.ndarray | torch.Tensor,
        k: int = 10,
        weight: float = 0.5,
    ) -> None:
        """
        Build the dataset and precompute mean kNN optimal labels.

        Args:
            model: an instance of optModel
            feats: data features
            costs: costs of objective function
            k: number of nearest neighbours selected
            weight: self-weight in the kNN convex combination (1.0 = no smoothing)
        """
        _validate_inputs(model, feats, costs)
        self.model = model
        # at most num_data-1 neighbours exist (self excluded), so k must stay below it
        num_data = len(feats)
        validate_positive_int(k, "k")
        if k >= num_data:
            raise ValueError(f"Invalid k={k}; must satisfy 1 <= k < num_data ({num_data}).")
        validate_probability(weight, "weight")
        # kNN loss parameters
        self.k = k
        self.weight = weight
        # data
        self.feats = feats
        self.costs = costs
        # find optimal solutions
        sols, objs = self._get_sols()
        self.feats = _as_float_tensor(self.feats)
        self.costs = _as_float_tensor(self.costs)
        self.sols = _as_float_tensor(sols)
        self.objs = _as_float_tensor(objs)

    def _get_sols(self) -> tuple[np.ndarray, np.ndarray]:
        """
        A method to get optimal solutions for all cost vectors
        """
        sols = []
        objs = []
        logger.info("Optimizing for optDataset...")
        # get kNN costs
        costs_knn = self._get_knn()
        # solve optimization
        for c_knn in tqdm(costs_knn):
            sol_knn = np.zeros((self.costs.shape[1], self.k))
            obj_knn = np.zeros(self.k)
            for i, c in enumerate(c_knn.T):
                sol_i, obj_i = self._solve(c)
                sol_knn[:, i] = _solution_to_numpy(sol_i)
                obj_knn[i] = obj_i
            # get average
            sol = sol_knn.mean(axis=1)
            obj = obj_knn.mean()
            sols.append(sol)
            objs.append(obj)
        # update cost as average kNN
        self.costs = costs_knn.mean(axis=2)
        return np.stack(sols), np.asarray(objs).reshape(-1, 1)

    def _get_knn(self) -> np.ndarray:
        """
        A method to get kNN costs
        """
        # scipy needs host numpy arrays
        if isinstance(self.feats, torch.Tensor):
            self.feats = self.feats.detach().cpu().numpy()
        if isinstance(self.costs, torch.Tensor):
            self.costs = self.costs.detach().cpu().numpy()
        # calculate distances between features
        distances = distance.cdist(self.feats, self.feats, "euclidean")
        # exclude self (diagonal) to get true nearest neighbours
        np.fill_diagonal(distances, np.inf)
        indexes = np.argpartition(distances, self.k, axis=1)[:, : self.k]
        # vectorized interpolation: (n, num_cost, 1) + (n, num_cost, k)
        neighbours = self.costs[indexes]  # (n, k, num_cost)
        costs_knn = self.weight * self.costs[:, :, np.newaxis] + (1 - self.weight) * np.transpose(
            neighbours, (0, 2, 1)
        )
        return costs_knn


class optDatasetConstrs(optDataset):
    """
    PyTorch ``Dataset`` for the CaVE cone-aligned loss.

    Stores features and cost coefficients, solves each instance, and extracts
    the **normals of the binding constraints at the optimal vertex** in
    canonical ``<=`` orientation.
    These normals span the polyhedral cone onto which ``coneAlignedCosine``
    projects the predicted cost vector during training.

    CaVE is defined for binary linear programs only, so the optimal vertex
    must be binary; instances that are infeasible or have non-binary optima
    raise (or are skipped when ``skip_infeas=True``). Binding-constraint
    extraction uses Gurobi's sparse-matrix API, which is why this dataset
    currently requires a Gurobi-backed ``optModel``.

    Per-instance row counts differ (different constraints bind at different
    vertices), so batches must be assembled with ``collate_tight_constraints``.

    Reference: Tang & Khalil (2024)
    `<https://link.springer.com/chapter/10.1007/978-3-031-60599-4_12>`_

    Attributes:
        model (optModel): Gurobi-backed optimization model
        feats (torch.Tensor): Data features
        costs (torch.Tensor): Cost vectors
        sols (torch.Tensor): Optimal solutions
        objs (torch.Tensor): Optimal objective values
        ctrs (list[torch.Tensor]): Per-instance binding-constraint normals
            in canonical ``<=`` orientation (ragged row counts)
    """

    def __init__(
        self,
        model: optModel,
        feats: np.ndarray | torch.Tensor,
        costs: np.ndarray | torch.Tensor,
        skip_infeas: bool = False,
    ) -> None:
        """
        Build the dataset and extract binding-constraint normals at each optimum.

        Args:
            model: an instance of optModel (Gurobi-backed)
            feats: data features
            costs: costs of objective function
            skip_infeas: if True, drop infeasible instances instead of raising
        """
        _validate_inputs(model, feats, costs)
        self.model = model
        self.skip_infeas = skip_infeas
        # data
        self.feats = feats
        self.costs = costs
        # find optimal solutions and binding constraints
        sols, objs, ctrs, valid = self._get_sols()
        # pre-convert to tensors (on CPU) to avoid repeated numpy→tensor copies
        self.feats = _as_float_tensor(self.feats[valid])
        self.costs = _as_float_tensor(self.costs[valid])
        self.sols = _as_float_tensor(sols)
        self.objs = _as_float_tensor(objs)
        self.ctrs = [_as_float_tensor(c) for c in ctrs]

    def _get_sols(  # type: ignore[override]
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[int]]:
        """
        A method to get optimal solutions and binding-constraint normals
        """
        # lazy gurobipy import (only optDatasetConstrs needs it)
        from gurobipy import GRB

        sols: list[np.ndarray] = []
        objs: list[list[float]] = []
        ctrs: list[np.ndarray] = []
        valid: list[int] = []
        logger.info("Optimizing for optDatasetConstrs...")
        model = self.model
        for i, c in enumerate(tqdm(self.costs)):
            try:
                sol, obj = self._solve(c)
            except RuntimeError as e:
                if self.skip_infeas:
                    logger.warning("Instance %d had no solution, skipping: %s", i, e)
                    continue
                raise
            # infeasibility check
            if model._model.Status != GRB.OPTIMAL:
                if self.skip_infeas:
                    logger.warning(
                        "Instance %d non-optimal (Status=%d), skipping.",
                        i,
                        model._model.Status,
                    )
                    continue
                raise ValueError(
                    f"Instance {i} did not solve to optimality "
                    f"(Gurobi Status={model._model.Status})."
                )
            # binary-vertex check: CaVE is defined for binary linear programs
            sol_arr = np.asarray(sol, dtype=np.float64)
            is_binary = np.all(
                np.isclose(sol_arr, 0.0, atol=1e-5) | np.isclose(sol_arr, 1.0, atol=1e-5)
            )
            if not is_binary:
                if self.skip_infeas:
                    logger.warning("Instance %d optimal vertex is not binary, skipping.", i)
                    continue
                raise ValueError(
                    f"Instance {i} optimal vertex is not binary; "
                    "CaVE requires binary linear programs."
                )
            sols.append(sol_arr)
            objs.append([float(obj)])
            ctrs.append(_extract_tight_normals(model, sol_arr))
            valid.append(i)
        if not valid:
            raise ValueError("No valid instances (all skipped or empty input).")
        return np.stack(sols), np.asarray(objs), ctrs, valid

    def __getitem__(  # type: ignore[override]
        self,
        index: int,
    ) -> tuple[torch.Tensor, ...]:
        """
        A method to retrieve data
        """
        return (
            cast("torch.Tensor", self.feats[index]),
            cast("torch.Tensor", self.costs[index]),
            self.sols[index],
            self.objs[index],
            self.ctrs[index],
        )


def collate_tight_constraints(batch):
    """
    Collate function for ``optDatasetConstrs`` batches.

    Stacks the standard ``(x, c, w, z)`` tensors and zero-pads the ragged
    per-instance binding-constraint matrices to a common row count so they
    can be assembled into a single ``(batch, max_rows, num_cost)`` tensor
    for ``coneAlignedCosine``.
    """
    from torch.nn.utils.rnn import pad_sequence

    x, c, w, z, t_ctrs = zip(*batch)
    return (
        torch.stack(x, dim=0),
        torch.stack(c, dim=0),
        torch.stack(w, dim=0),
        torch.stack(z, dim=0),
        pad_sequence(list(t_ctrs), batch_first=True, padding_value=0),
    )


def _extract_tight_normals(
    model: optModel,
    sol: np.ndarray,
    tol: float = 1e-5,
) -> np.ndarray:
    """
    A function to extract normals of binding constraints at sol in canonical <= orientation
    """
    import gurobipy as gp
    from gurobipy import GRB

    grb = model._model
    # TSP/VRP pre-cache _cost_vars; MVar / dict backends fall back to model.x
    cost_vars: list = model._cost_vars
    if not cost_vars:
        cost_vars = model.x.tolist() if isinstance(model.x, gp.MVar) else list(model.x.values())
    num_cost = len(cost_vars)
    sol_np = np.asarray(sol, dtype=np.float64)
    chunks: list[np.ndarray] = []
    # explicit constraints: batch slack + sense + vectorized sign flip
    constrs = grb.getConstrs()
    if constrs:
        slacks = np.asarray(grb.getAttr("Slack", constrs))
        senses_arr = np.asarray(grb.getAttr("Sense", constrs))
        tight_mask = np.abs(slacks) < tol
        if tight_mask.any():
            # cost-column slice cached across solves; the row count keys invalidation
            cache = getattr(grb, "_cave_A_cost", None)
            if cache is None or cache[0] != grb.NumConstrs:
                cost_col_idx = np.asarray([v.index for v in cost_vars])
                cache = (grb.NumConstrs, grb.getA().tocsr()[:, cost_col_idx])
                grb._cave_A_cost = cache
            # extract all tight rows in a single sparse-to-dense conversion
            A_tight = cache[1][tight_mask].toarray()
            tight_senses = senses_arr[tight_mask]
            is_le = tight_senses == GRB.LESS_EQUAL
            is_ge = tight_senses == GRB.GREATER_EQUAL
            is_eq = tight_senses == GRB.EQUAL
            if not (is_le | is_ge | is_eq).all():
                bad = tight_senses[~(is_le | is_ge | is_eq)][0]
                raise ValueError(f"Invalid constraint sense {bad!r}.")
            # <= kept as-is, >= negated, == contributes ± rows
            if is_le.any():
                chunks.append(A_tight[is_le])
            if is_ge.any():
                chunks.append(-A_tight[is_ge])
            if is_eq.any():
                chunks.append(A_tight[is_eq])
                chunks.append(-A_tight[is_eq])
    # lazy constraints: evaluate LHS at the optimum to derive slack
    var_to_cost: dict[str, int] = {v.VarName: k for k, v in enumerate(cost_vars)}
    lazy_rows: list[np.ndarray] = []
    for tc in getattr(grb, "_lazy_constrs", []):
        parsed = _parse_temp_constraint(tc, var_to_cost, num_cost)
        if parsed is None:
            continue
        coefs, rhs, sense = parsed
        lhs_val = float(coefs @ sol_np)
        if abs(rhs - lhs_val) < tol:
            lazy_rows.extend(_orient_constraint_row(coefs, sense))
    if lazy_rows:
        chunks.append(np.asarray(lazy_rows))
    # binary variable bounds: vectorized via masks (mutually exclusive)
    low_mask = sol_np <= tol
    high_mask = (sol_np >= 1 - tol) & ~low_mask
    n_low = int(low_mask.sum())
    n_high = int(high_mask.sum())
    # tight at 0: -e_k rows
    if n_low > 0:
        low_rows = np.zeros((n_low, num_cost), dtype=np.float64)
        low_rows[np.arange(n_low), np.where(low_mask)[0]] = -1.0
        chunks.append(low_rows)
    # tight at 1: +e_k rows
    if n_high > 0:
        high_rows = np.zeros((n_high, num_cost), dtype=np.float64)
        high_rows[np.arange(n_high), np.where(high_mask)[0]] = 1.0
        chunks.append(high_rows)
    # empty fallback
    if not chunks:
        return np.zeros((0, num_cost), dtype=np.float32)
    return np.vstack(chunks).astype(np.float32)


def _orient_constraint_row(row: np.ndarray, sense: str) -> list[np.ndarray]:
    """Return constraint rows in canonical ``<=`` orientation."""
    from gurobipy import GRB

    # <=
    if sense == GRB.LESS_EQUAL:
        return [row]
    # >= negated to <=
    if sense == GRB.GREATER_EQUAL:
        return [-row]
    # == split into <= and >=
    if sense == GRB.EQUAL:
        return [row, -row]
    raise ValueError(f"Invalid constraint sense {sense!r}.")


def _parse_temp_constraint(
    tc,
    var_to_cost: dict[str, int],
    num_cost: int,
) -> tuple[np.ndarray, float, str] | None:
    """
    Parse a Gurobi TempConstr into (coefs, rhs, sense) over the cost-vector dim
    """
    # TempConstr internals
    lhs = getattr(tc, "_lhs", None)
    rhs = getattr(tc, "_rhs", None)
    sense = getattr(tc, "_sense", None)
    # unparseable fallback
    if lhs is None or rhs is None or sense is None:
        return None
    # project LinExpr terms onto cost-vector dim
    coefs = np.zeros(num_cost, dtype=np.float64)
    for i in range(lhs.size()):
        var = lhs.getVar(i)
        k = var_to_cost.get(var.VarName)
        if k is not None:
            coefs[k] += lhs.getCoeff(i)
    return coefs, float(rhs), sense
