#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from model import optModel

class optDataset(Dataset):
    """optimization problem Dataset"""

    def __init__(self, model, feats, costs):
        """
        Args:
            model: an instance of optModel
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
        print('Solve optimization problems...')
        time.sleep(1)
        for c in tqdm(self.c):
            sol, obj = self.solve(c)
            sols.append(sol)
            objs.append(obj)
        return np.array(sols), np.array(objs)

    def solve(self, cost):
        """
        solve optimization problem to get optimal solutions with given cost
        """
        self.model.setObj(cost)
        sol = self.model.solve()
        return sol

    def __len__(self):
        """
        return the number of optimization problems
        """
        return len(self.c)

    def __getitem__(self, index):
        """
        return feature, cost and optimal solution
        """
        return torch.from_numpy(self.x[index]), \
               torch.from_numpy(self.c[index]), \
               torch.from_numpy(self.w[index]), \
               torch.from_numpy(self.z[index])
