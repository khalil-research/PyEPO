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

from pyepo.model.opt import optModel

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
        # pre-convert to tensors (on CPU) to avoid repeated numpy→tensor copies
        self.feats = torch.as_tensor(feats, dtype=torch.float32)
        self.costs = torch.as_tensor(costs, dtype=torch.float32)
        self.sols = torch.as_tensor(sols, dtype=torch.float32)
        self.objs = torch.as_tensor(objs, dtype=torch.float32)

    def _getSols(self) -> tuple[np.ndarray, np.ndarray]:
        """
        A method to get optimal solutions for all cost vectors
        """
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
            sols.append(sol)
            objs.append([obj])
        return np.array(sols), np.array(objs)

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
        feats (np.ndarray): Data features
        costs (np.ndarray): Cost vectors
        sols (np.ndarray): Optimal solutions
        objs (np.ndarray): Optimal objective values
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
        # pre-convert to tensors (on CPU) to avoid repeated numpy→tensor copies
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
            objs.append([obj])
        # update cost as average kNN
        self.costs = costs_knn.mean(axis=2)
        return np.array(sols), np.array(objs)

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
