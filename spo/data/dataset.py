#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from spo.model import optModel

class optDataset(Dataset):
    """optimization problem Dataset"""

    def __init__(self, model, feats, costs):
        """
        Args:
            model: an instance of optModel
            feats: data features
            costs: data labels, costs of objective function
        """
        assert isinstance(model, optModel), 'arg model is not an optModel'
        self.model = model
        # data
        self.x = feats
        self.c = costs
        # find optimal solutions
        self.w, self.z = self.getSols()

    def getSols(self):
        """
        get optimal solutions for all cost vectors
        """
        sols = []
        objs = []
        print('Optimizing for optDataset...')
        time.sleep(1)
        for c in tqdm(self.c):
            try:
                sol, obj = self._solve(c)
            except:
                raise ValueError("For optModel, the method 'solve' should return solution vector and objective value.")
            sols.append(sol)
            objs.append([obj])
        return np.array(sols), np.array(objs)

    def _solve(self, cost):
        """
        solve optimization problem to get optimal solutions with given cost
        Args:
            costs: cost of objective function
        """
        self.model.setObj(cost)
        sol, obj = self.model.solve()
        return sol, obj

    def __len__(self):
        """
        return the number of optimization problems
        """
        return len(self.c)

    def __getitem__(self, index):
        """
        return feature, cost and optimal solution
        """
        return torch.FloatTensor(self.x[index]), \
               torch.FloatTensor(self.c[index]), \
               torch.FloatTensor(self.w[index]), \
               torch.FloatTensor(self.z[index])
