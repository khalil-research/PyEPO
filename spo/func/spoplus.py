#!/usr/bin/env python
# coding: utf-8

import inspect
import torch
from torch.autograd import Function
import numpy as np
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool
import spo
from spo.model import optModel

def solveWithObj4Par(cost, args, model_name):
    """
    global solve function for parallel
    """
    # build model
    try:
        model = eval(model_name)(**args)
    except:
        model = eval('spo.model.{}'.format(model_name))(**args)
    # set obj
    model.setObj(cost)
    # solve
    sol, obj = model.solve()
    return sol, obj


def getArgs(model):
    """
    get args of model
    """
    for mem in inspect.getmembers(model):
        if mem[0] == '__dict__':
            attrs = mem[1]
            args = {}
            for name in attrs:
                if name in inspect.signature(model.__init__).parameters:
                    args[name] = attrs[name]
            return args


class SPOPlus(Function):
    """
    SPO+ Loss function, a surrogate loss function of SPO Loss, which
    measure the decision error (optimality gap) of optimization problem.

    For SPO/SPO+ Loss, the objective function is linear and constraints
    are known and fixed, but the cost vector need to be predicted from
    contextual data.

    The SPO+ Loss is convex with subgradient. Thus, allows us to design
    an algorithm based on stochastic gradient descent.
    """
    def __init__(self, model, processes=1):
        """
        args:
          processes: number of processors, 1 for single-core, 0 for number of CPUs
        """
        super().__init__()
        # optimization model
        assert isinstance(model, optModel), 'arg model is not an optModel'
        global _SPO_FUNC_SPOP_OPTMODEL
        _SPO_FUNC_SPOP_OPTMODEL = model
        # num of processors
        assert processes in range(mp.cpu_count()), IndexError('Invalid processors number.')
        global _SPO_FUNC_SPOP_PROCESSES
        _SPO_FUNC_SPOP_PROCESSES = processes
        print('Num of cores: {}'.format(_SPO_FUNC_SPOP_PROCESSES))

    @staticmethod
    def forward(ctx, pred_cost, true_cost, true_sol, true_obj):
        """
        args:
          model: optModel
          pred_cost: predicted costs
          true_cost: true costs
          true_sol: true solutions
          true_obj: true objective values
        """
        # get device
        device = pred_cost.device
        # get global
        model = _SPO_FUNC_SPOP_OPTMODEL
        processes = _SPO_FUNC_SPOP_PROCESSES
        # convert tenstor
        cp = pred_cost.to('cpu').numpy()
        c = true_cost.to('cpu').numpy()
        w = true_sol.to('cpu').numpy()
        z = true_obj.to('cpu').numpy()
        # check sol
        ins_num = len(z)
        for i in range(ins_num):
            assert abs(z[i] - np.dot(c[i], w[i])) / (abs(z[i]) + 1e-3) < 1e-3, \
            'Solution {} does not macth the objective value {}.'.format(np.dot(c[i], w[i]), z[i][0])
        # single-core
        if processes == 1:
            loss = []
            sol = []
            for i in range(ins_num):
                # solve
                model.setObj(2*cp[i]-c[i])
                solq, objq = model.solve()
                # calculate loss
                loss.append(-objq+2*np.dot(cp[i],w[i])-z[i])
                # solution
                sol.append(solq)
        # multi-core
        else:
            # get class
            model_name = type(model).__name__
            # get args
            args = getArgs(model)
            # number of processes
            processes = mp.cpu_count() if not processes else processes
            # parallel computing
            with ProcessingPool(processes=processes) as pool:
                res = pool.amap(solveWithObj4Par, 2*cp-c, [args]*ins_num, [model_name]*ins_num).get()
            # get res
            sol = np.array(list(map(lambda x: x[0], res)))
            obj = np.array(list(map(lambda x: x[1], res)))
            # calculate loss
            loss = []
            for i in range(ins_num):
                loss.append(-obj[i]+2*np.dot(cp[i],w[i])-z[i])
        # convert to tensor
        loss = torch.FloatTensor(loss).to(device)
        sol = torch.FloatTensor(sol).to(device)
        # save solutions
        ctx.save_for_backward(true_sol, sol)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        w, wq = ctx.saved_tensors
        grad = 2 * (w - wq).mean(0)
        return grad_output * grad, None, None, None
