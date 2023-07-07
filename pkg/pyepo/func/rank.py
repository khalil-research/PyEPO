"""
Learning To Rank Loss functions
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pyepo import EPO
from pyepo.func.abcmodule import optModule
from pyepo.data.dataset import optDataset
from pyepo.func.utlis import _solveWithObj4Par, _solve_in_pass, _cache_in_pass


class listwiseLTR(optModule):
    """
        An autograd module for the listwise learning to rank loss.
        For the listwise learning to rank loss, the constraints are known and fixed,
        but the cost vector needs to be predicted from contextual data.
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, dataset)
        # solution pool
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        self.solpool = dataset.sols.copy()

    def forward(self, pred_cost, true_cost, reduction="mean"):
        """
        Forward pass
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach().to("cpu").numpy()
        # solve
        if np.random.uniform() <= self.solve_ratio:
            sol, _ = _solve_in_pass(cp, self.optmodel, self.processes, self.pool)
            self.solpool = np.concatenate((self.solpool, sol))
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # get loss
        objpool_c = true_cost @ solpool.T
        objpool_cp = pred_cost @ solpool.T
        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss = - (F.log_softmax(objpool_cp, dim=1) *
                      F.softmax(objpool_c, dim=1))
        if self.optmodel.modelSense == EPO.MAXIMIZE:
            loss = - (F.log_softmax(- objpool_cp, dim=1) *
                      F.softmax(- objpool_c, dim=1))
        # reduction
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(reduction))
        return loss


class pairwiseLTR(optModule):
    """
        An autograd module for the pairwise learning to rank loss.
        For the pairwise learning to rank loss, the constraints are known and fixed,
        but the cost vector needs to be predicted from contextual data.
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, dataset)
        # solution pool
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        self.solpool = dataset.sols.copy()

    def forward(self, pred_cost, true_cost, reduction="mean"):
        """
        Forward pass
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach().to("cpu").numpy()
        # solve
        if np.random.uniform() <= self.solve_ratio:
            sol, _ = _solve_in_pass(cp, self.optmodel, self.processes, self.pool)
            self.solpool = np.concatenate((self.solpool, sol))
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # get loss
        loss = []
        relu = nn.ReLU()
        for i in range(len(pred_cost)):
            objpool_c_i = torch.matmul(true_cost[i], solpool.T)
            objpool_cp_i = torch.matmul(pred_cost[i], solpool.T)
            _, indices = np.unique(objpool_c_i.detach().numpy(), return_index=True)
            if self.optmodel.modelSense == EPO.MINIMIZE:
                indices = indices[::-1]
            big_ind = [indices[0] for _ in range(len(indices) - 1)]
            small_ind = [indices[p + 1] for p in range(len(indices) - 1)]
            if self.optmodel.modelSense == EPO.MINIMIZE:
                loss.append(relu(objpool_cp_i[small_ind] - objpool_cp_i[big_ind]))
            if self.optmodel.modelSense == EPO.MAXIMIZE:
                loss.append(relu(objpool_cp_i[big_ind] - objpool_cp_i[small_ind]))
        loss = torch.stack(loss)
        # reduction
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(reduction))
        return loss


class pointwiseLTR(optModule):
    """
        An autograd module for the pointwise learning to rank loss.
        For the pointwise learning to rank loss, the constraints are known and fixed,
        but the cost vector needs to be predicted from contextual data.
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, dataset)
        # solution pool
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        self.solpool = dataset.sols.copy()

    def forward(self, pred_cost, true_cost, reduction="mean"):
        """
        Forward pass
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach().to("cpu").numpy()
        # solve
        if np.random.uniform() <= self.solve_ratio:
            sol, _ = _solve_in_pass(cp, self.optmodel, self.processes, self.pool)
            self.solpool = np.concatenate((self.solpool, sol))
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # get loss
        objpool_c = torch.matmul(true_cost, solpool.T)
        objpool_cp = torch.matmul(pred_cost, solpool.T)
        loss = (objpool_c - objpool_cp).square()
        # reduction
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(reduction))
        return loss
