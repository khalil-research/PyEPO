#!/usr/bin/env python
"""
optDataset class based on PyTorch Dataset
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import torch
from scipy.spatial import distance
from torch.utils.data import Dataset
from tqdm import tqdm

from pyepo import EPO
from pyepo.model.opt import optModel

try:
    from pyepo.model.mpax import optMpaxModel
except ImportError:
    optMpaxModel = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class optDataset(Dataset):
    """
    This class is a Torch Dataset for optimization problems.

    Attributes:
        model (optModel): Optimization model
        feats (torch.Tensor): Data features
        costs (torch.Tensor): Cost vectors
        sols (torch.Tensor): Optimal solutions
        objs (torch.Tensor): Optimal objective values
    """

    def __init__(
        self,
        model: optModel,
        feats: np.ndarray | torch.Tensor,
        costs: np.ndarray | torch.Tensor,
    ) -> None:
        """
        A method to create an optDataset from optModel

        Args:
            model: an instance of optModel
            feats: data features
            costs: costs of objective function
        """
        if not isinstance(model, optModel):
            raise TypeError("arg model is not an optModel")
        self.model = model
        # data
        self.feats = feats
        self.costs = costs
        # find optimal solutions
        sols, objs = self._getSols()
        self.feats = torch.as_tensor(feats, dtype=torch.float32)
        self.costs = torch.as_tensor(costs, dtype=torch.float32)
        self.sols = torch.as_tensor(sols, dtype=torch.float32)
        self.objs = torch.as_tensor(objs, dtype=torch.float32)

    def _getSols(self) -> tuple[np.ndarray, np.ndarray]:
        """
        A method to get optimal solutions for all cost vectors
        """
        # MPAX fast path: vmap-solve the whole dataset in a single dispatch
        if optMpaxModel is not None and isinstance(self.model, optMpaxModel):
            return self._getSolsMpaxBatch()
        sols = []
        objs = []
        logger.info("Optimizing for optDataset...")
        for c in tqdm(self.costs):
            try:
                sol, obj = self._solve(c)
                # to numpy
                if isinstance(sol, torch.Tensor):
                    sol = sol.detach().cpu().numpy()
            except Exception as e:
                raise ValueError(
                    "For optModel, the method 'solve' should return solution vector and objective value."
                ) from e
            sols.append(np.asarray(sol))
            objs.append(obj)
        return np.stack(sols), np.asarray(objs).reshape(-1, 1)

    def _getSolsMpaxBatch(self) -> tuple[np.ndarray, np.ndarray]:
        """
        A method to batch-solve every cost vector in one MPAX vmap call.
        """
        logger.info("Optimizing for optDataset (MPAX batched)...")
        self.model.setObj(self.costs)
        sols, objs = self.model.batch_optimize(self.model.c)
        # writable copy; torch.as_tensor warns on JAX read-only buffers
        sols_np = np.array(sols, dtype=np.float32)
        objs_np = np.array(objs, dtype=np.float32)
        # jitted_solve returns c·sol where setObj already negated c for MAX
        if self.model.modelSense == EPO.MAXIMIZE:
            objs_np = -objs_np
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
        self.model.setObj(cost)
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
    This class is a Torch Dataset for optimization problems, when using the robust kNN-loss.

    Reference: <https://arxiv.org/abs/2310.04328>

    Attributes:
        model (optModel): Optimization model
        k (int): number of nearest neighbours selected
        weight (float): weight of kNN-loss
        feats (torch.Tensor): Data features
        costs (torch.Tensor): Cost vectors
        sols (torch.Tensor): Optimal solutions
        objs (torch.Tensor): Optimal objective values
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
        A method to create an optDataset from optModel

        Args:
            model: an instance of optModel
            feats: data features
            costs: costs of objective function
            k: number of nearest neighbours selected
            weight: weight of kNN-loss
        """
        if not isinstance(model, optModel):
            raise TypeError("arg model is not an optModel")
        self.model = model
        # kNN loss parameters
        self.k = k
        self.weight = weight
        # data
        self.feats = feats
        self.costs = costs
        # find optimal solutions
        sols, objs = self._getSols()
        self.feats = torch.as_tensor(self.feats, dtype=torch.float32)
        self.costs = torch.as_tensor(self.costs, dtype=torch.float32)
        self.sols = torch.as_tensor(sols, dtype=torch.float32)
        self.objs = torch.as_tensor(objs, dtype=torch.float32)

    def _getSols(self) -> tuple[np.ndarray, np.ndarray]:
        """
        A method to get optimal solutions for all cost vectors
        """
        sols = []
        objs = []
        logger.info("Optimizing for optDataset...")
        # get kNN costs
        costs_knn = self._getKNN()
        # solve optimization
        for c_knn in tqdm(costs_knn):
            sol_knn = np.zeros((self.costs.shape[1], self.k))
            obj_knn = np.zeros(self.k)
            for i, c in enumerate(c_knn.T):
                try:
                    sol_i, obj_i = self._solve(c)
                    if isinstance(sol_i, torch.Tensor):
                        sol_i = sol_i.detach().cpu().numpy()
                except Exception as e:
                    raise ValueError(
                        "For optModel, the method 'solve' should return solution vector and objective value."
                    ) from e
                sol_knn[:, i] = sol_i
                obj_knn[i] = obj_i
            # get average
            sol = sol_knn.mean(axis=1)
            obj = obj_knn.mean()
            sols.append(sol)
            objs.append(obj)
        # update cost as average kNN
        self.costs = costs_knn.mean(axis=2)
        return np.stack(sols), np.asarray(objs).reshape(-1, 1)

    def _getKNN(self) -> np.ndarray:
        """
        A method to get kNN costs
        """
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
    This class is a Torch Dataset for optimization problems with the normals
    of binding constraints at the optimum, consumed by the CaVE loss.

    Reference: <https://link.springer.com/chapter/10.1007/978-3-031-60599-4_12>

    Attributes:
        model (optModel): Optimization model (Gurobi-backed)
        feats (torch.Tensor): Data features
        costs (torch.Tensor): Cost vectors
        sols (torch.Tensor): Optimal solutions
        objs (torch.Tensor): Optimal objective values
        ctrs (list[torch.Tensor]): Per-instance binding-constraint normals
    """

    def __init__(
        self,
        model: optModel,
        feats: np.ndarray | torch.Tensor,
        costs: np.ndarray | torch.Tensor,
        skip_infeas: bool = False,
    ) -> None:
        """
        A method to create an optDatasetConstrs from optModel

        Args:
            model: an instance of optModel
            feats: data features
            costs: costs of objective function
            skip_infeas: if True, drop infeasible instances instead of raising
        """
        if not isinstance(model, optModel):
            raise TypeError("arg model is not an optModel")
        self.model = model
        self.skip_infeas = skip_infeas
        # data
        self.feats = feats
        self.costs = costs
        # find optimal solutions and binding constraints
        sols, objs, ctrs, valid = self._getSols()
        # pre-convert to tensors (on CPU) to avoid repeated numpy→tensor copies
        self.feats = torch.as_tensor(self.feats[valid], dtype=torch.float32)
        self.costs = torch.as_tensor(self.costs[valid], dtype=torch.float32)
        self.sols = torch.as_tensor(sols, dtype=torch.float32)
        self.objs = torch.as_tensor(objs, dtype=torch.float32)
        self.ctrs = [torch.as_tensor(c, dtype=torch.float32) for c in ctrs]

    def _getSols(  # type: ignore[override]
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
        for i, c in enumerate(tqdm(self.costs)):
            # fresh per-instance copy keeps the lazy-constraint buffer clean
            model = self.model.copy()
            model.setObj(c)
            sol, obj = model.solve()
            # infeasibility check
            if model._model.Status != GRB.OPTIMAL:
                if self.skip_infeas:
                    logger.warning(
                        "Instance %d non-optimal (Status=%d), skipping.",
                        i, model._model.Status,
                    )
                    continue
                raise ValueError(
                    f"Instance {i} did not solve to optimality "
                    f"(Gurobi Status={model._model.Status})."
                )
            # binary-vertex check: CaVE is defined for binary linear programs
            sol_arr = np.asarray(sol, dtype=np.float64)
            if not np.all((sol_arr < 1e-5) | (sol_arr > 1 - 1e-5)):
                raise ValueError(
                    f"Instance {i} optimal vertex is not binary; "
                    "CaVE requires binary linear programs."
                )
            sols.append(sol_arr)
            objs.append([float(obj)])
            ctrs.append(_extract_tight_normals(model, sol))
            valid.append(i)
        return np.stack(sols), np.asarray(objs), ctrs, valid

    def __len__(self) -> int:
        """
        A method to get data size
        """
        return len(self.feats)

    def __getitem__(  # type: ignore[override]
        self, index: int,
    ) -> tuple[torch.Tensor, ...]:
        """
        A method to retrieve data
        """
        return (
            self.feats[index],
            self.costs[index],
            self.sols[index],
            self.objs[index],
            self.ctrs[index],
        )


def collate_tight_constraints(batch):
    """
    A custom collate function for PyTorch DataLoader that pads binding-constraint matrices
    """
    from torch.nn.utils.rnn import pad_sequence
    x, c, w, z, t_ctrs = zip(*batch)
    return (
        torch.stack(x, dim=0),
        torch.stack(c, dim=0),
        torch.stack(w, dim=0),
        torch.stack(z, dim=0),
        pad_sequence(t_ctrs, batch_first=True, padding_value=0),
    )


def _extract_tight_normals(
    model: optModel, sol: np.ndarray, tol: float = 1e-5,
) -> np.ndarray:
    """
    A function to extract normals of binding constraints at sol in canonical <= orientation
    """
    from gurobipy import GRB
    grb = model._model
    cost_vars: list = model._cost_vars
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
            # project the constraint matrix onto cost-variable columns
            cost_col_idx = np.asarray([v.index for v in cost_vars])
            A = grb.getA().tocsr()
            # extract all tight rows in a single sparse-to-dense conversion
            A_tight = A[:, cost_col_idx][tight_mask].toarray()
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
        coefs, rhs, sense = _parse_temp_constraint(tc, var_to_cost, num_cost)
        if coefs is None:
            continue
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
    tc, var_to_cost: dict[str, int], num_cost: int,
) -> tuple[np.ndarray | None, float | None, str | None]:
    """
    Parse a Gurobi TempConstr into (coefs, rhs, sense) over the cost-vector dim
    """
    # TempConstr internals
    lhs = getattr(tc, "_lhs", None)
    rhs = getattr(tc, "_rhs", None)
    sense = getattr(tc, "_sense", None)
    # unparseable fallback
    if lhs is None or rhs is None or sense is None:
        return None, None, None
    # project LinExpr terms onto cost-vector dim
    coefs = np.zeros(num_cost, dtype=np.float64)
    for i in range(lhs.size()):
        var = lhs.getVar(i)
        k = var_to_cost.get(var.VarName)
        if k is not None:
            coefs[k] += lhs.getCoeff(i)
    return coefs, float(rhs), sense
