"""
Learning To Rank Loss functions
"""

import multiprocessing as mp
from abc import abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from pathos.multiprocessing import ProcessingPool
from torch import nn

from pyepo.data.dataset import optDataset
from pyepo.model.opt import optModel

from pyepo.func.utlis import _solveWithObj4Par, _solve_in_pass, _cache_in_pass


class learningToRank(nn.Module):
    """
        An abstract module for the learning to rank losses, which measure the difference in how the predicted cost
        vector and the true cost vector rank a pool of feasible solutions.
    """
    def __init__(self, optmodel, processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__()
        # optimization model
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel")
        self.optmodel = optmodel
        # number of processes
        if processes not in range(mp.cpu_count()+1):
            raise ValueError("Invalid processors number {}, only {} cores.".
                format(processes, mp.cpu_count()))
        self.processes = mp.cpu_count() if not processes else processes
        # single-core
        if processes == 1:
            self.pool = None
        # multi-core
        else:
            self.pool = ProcessingPool(processes)
        print("Num of cores: {}".format(self.processes))
        # solution pool
        self.solve_ratio = solve_ratio
        if (self.solve_ratio < 0) or (self.solve_ratio > 1):
            raise ValueError("Invalid solving ratio {}. It should be between 0 and 1.".
                format(self.solve_ratio))
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        self.solpool = dataset.sols.copy()

    @abstractmethod
    def forward(self, pred_cost, true_cost, reduction="mean"):
        """
        Forward pass
        """
        # convert tensor
        pass


class listwiseLTR(learningToRank):
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
        solpool_obj_c = torch.matmul(true_cost, solpool.T)
        solpool_obj_cp = torch.matmul(pred_cost, solpool.T)
        loss = -(F.log_softmax(-self.optmodel.modelSense * solpool_obj_cp, dim=1) *
                  F.softmax(-self.optmodel.modelSense * solpool_obj_c, dim=1))
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


class pairwiseLTR(learningToRank):
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
        loss = 0
        relu = nn.ReLU()
        for i in range(len(pred_cost)):
            solpool_obj_c_i = torch.matmul(true_cost[i], solpool.T)
            solpool_obj_cp_i = torch.matmul(pred_cost[i], solpool.T)
            _, indices = np.unique((self.optmodel.modelSense * solpool_obj_c_i).detach().numpy(), return_index=True)
            big_ind = [indices[0] for _ in range(len(indices) - 1)]
            small_ind = [indices[p + 1] for p in range(len(indices) - 1)]
            loss += relu(self.optmodel.modelSense * (solpool_obj_cp_i[big_ind] - solpool_obj_cp_i[small_ind]))
        loss /= len(pred_cost)
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


class pointwiseLTR(learningToRank):
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
        solpool_obj_c = torch.matmul(true_cost, solpool.T)
        solpool_obj_cp = torch.matmul(pred_cost, solpool.T)
        loss = (solpool_obj_c - solpool_obj_cp).square()
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
