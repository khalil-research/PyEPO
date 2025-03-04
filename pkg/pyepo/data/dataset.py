#!/usr/bin/env python
# coding: utf-8
"""
optDataset class based on PyTorch Dataset
"""

import time

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from pyepo.model.opt import optModel

import random
from scipy.spatial import distance

class optDataset(Dataset):
    """
    This class is Torch Dataset for optimization problems.

    Attributes:
        model (optModel): Optimization models
        feats (np.ndarray): Data features
        costs (np.ndarray): Cost vectors
        sols (np.ndarray): Optimal solutions
        objs (np.ndarray): Optimal objective values
    """

    def __init__(self, model, feats, costs):
        """
        A method to create a optDataset from optModel

        Args:
            model (optModel): an instance of optModel
            feats (np.ndarray): data features
            costs (np.ndarray): costs of objective function
        """
        if not isinstance(model, optModel):
            raise TypeError("arg model is not an optModel")
        self.model = model
        # data
        self.feats = feats
        self.costs = costs
        # find optimal solutions
        self.sols, self.objs = self._getSols()

    def _getSols(self):
        """
        A method to get optimal solutions for all cost vectors
        """
        sols = []
        objs = []
        print("Optimizing for optDataset...")
        time.sleep(1)
        for c in tqdm(self.costs):
            try:
                sol, obj = self._solve(c)
                # to numpy
                if isinstance(sol, torch.Tensor):
                    sol = sol.detach().cpu().numpy()
            except:
                raise ValueError(
                    "For optModel, the method 'solve' should return solution vector and objective value."
                )
            sols.append(sol)
            objs.append([obj])
        return np.array(sols), np.array(objs)

    def _solve(self, cost):
        """
        A method to solve optimization problem to get an optimal solution with given cost

        Args:
            cost (np.ndarray): cost of objective function

        Returns:
            tuple: optimal solution (np.ndarray) and objective value (float)
        """
        self.model.setObj(cost)
        sol, obj = self.model.solve()
        return sol, obj

    def __len__(self):
        """
        A method to get data size

        Returns:
            int: the number of optimization problems
        """
        return len(self.costs)

    def __getitem__(self, index):
        """
        A method to retrieve data

        Args:
            index (int): data index

        Returns:
            tuple: data features (torch.tensor), costs (torch.tensor), optimal solutions (torch.tensor) and objective values (torch.tensor)
        """
        return (
            torch.FloatTensor(self.feats[index]),
            torch.FloatTensor(self.costs[index]),
            torch.FloatTensor(self.sols[index]),
            torch.FloatTensor(self.objs[index]),
        )


class optDatasetKNN(optDataset):
    """
    This class is Torch Dataset for optimization problems, when using the robust kNN-loss.

    Reference: <https://arxiv.org/abs/2310.04328>

    Attributes:
        model (optModel): Optimization models
        k (int): number of nearest neighbours selected
        weight (float): weight of kNN-loss
        feats (np.ndarray): Data features
        costs (np.ndarray): Cost vectors
        sols (np.ndarray): Optimal solutions
        objs (np.ndarray): Optimal objective values
    """
    def __init__(self, model, feats, costs, k=10, weight=0.5):
        """
        A method to create a optDataset from optModel

        Args:
            model (optModel): an instance of optModel
            feats (np.ndarray): data features
            costs (np.ndarray): costs of objective function
            k (int): number of nearest neighbours selected
            weight (float): weight of kNN-loss
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
        self.sols, self.objs = self._getSols()

    def _getSols(self):
        """
        A method to get optimal solutions for all cost vectors
        """
        sols = []
        objs = []
        print("Optimizing for optDataset...")
        time.sleep(1)
        # get kNN costs
        costs_knn = self._getKNN()
        # solve optimization
        for c_knn in tqdm(costs_knn):
            sol_knn = np.zeros((self.costs.shape[1], self.k))
            obj_knn = np.zeros(self.k)
            for i, c in enumerate(c_knn.T):
                try:
                    sol_i, obj_i = self._solve(c)
                except:
                    raise ValueError(
                        "For optModel, the method 'solve' should return solution vector and objective value."
                    )
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

    def _getKNN(self):
        """
        A method to get kNN costs
        """
        # init costs
        costs_knn = np.zeros((*self.costs.shape, self.k))
        # calculate distances between features
        distances = distance.cdist(self.feats, self.feats, "euclidean")
        indexes = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        # get neighbours costs
        for i, knns in enumerate(indexes):
            # interpolation weight
            costs_knn[i] = self.weight * self.costs[i].reshape((-1, 1)) \
                         + (1 - self.weight) * self.costs[knns].T
        return costs_knn
