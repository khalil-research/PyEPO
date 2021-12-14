#!/usr/bin/env python
# coding: utf-8
"""
Torch Dataset for optimization
"""

import time

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from spo.model import optModel


class optDataset(Dataset):
    """
    This class is Torch Dataset for optimization problems.

    Attributes:
        m (optModel): an instance of optModel
        x (ndarray): data features
        c (ndarray): costs of objective function
        w (ndarray): optimal solutions
        z (ndarray): optimal objective values
    """

    def __init__(self, model, feats, costs):
        """
        A method to create a optDataset from optModel

        Args:
            model (optModel): an instance of optModel
            feats (ndarray): data features
            costs (ndarray): costs of objective function
        """
        if not isinstance(model, optModel):
            raise TypeError("arg model is not an optModel")
        self.model = model
        # data
        self.x = feats
        self.c = costs
        # find optimal solutions
        self.w, self.z = self._getSols()

    def _getSols(self):
        """
        A method to get optimal solutions for all cost vectors
        """
        sols = []
        objs = []
        print("Optimizing for optDataset...")
        time.sleep(1)
        for c in tqdm(self.c):
            try:
                sol, obj = self._solve(c)
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
            cost (ndarray): cost of objective function

        Returns:
            tuple: optimal solution (ndarray) and objective value (float)
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
        return len(self.c)

    def __getitem__(self, index):
        """
        A method to retrieve data

        Args:
            index (int): data index

        Returns:
            tuple: data features (tensor), costs (tensor), optimal solutions (tensor) and objective values (tensor)
        """
        return (
            torch.FloatTensor(self.x[index]),
            torch.FloatTensor(self.c[index]),
            torch.FloatTensor(self.w[index]),
            torch.FloatTensor(self.z[index]),
        )
