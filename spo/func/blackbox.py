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
    sol, _ = model.solve()
    return sol


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


class blackboxOpt(Function):
    """
    block-box optimizer function, which is diffenretiable to introduce
    blocks into neural networks.

    For block-box, the objective function is linear and constraints are
    known and fixed, but the cost vector need to be predicted from
    contextual data.

    The block-box approximate gradient of optimizer smoothly. Thus,
    allows us to design an algorithm based on stochastic gradient
    descent.
    """
    def __init__(self, processes=1):
        """
        args:
          processes: number of processors, 0 for single-core, -1 for number of CPUs
        """
        super().__init__()
        # num of processors
        assert processes in range(mp.cpu_count()), IndexError('Invalid processors number.')
        global _SPO_FUNC_BB_PROCESSES
        _SPO_FUNC_BB_PROCESSES = processes
        print('Num of cores: {}'.format(_SPO_FUNC_BB_PROCESSES))

    @staticmethod
    def forward(ctx, model, pred_cost, lambd=10):
        """
        args:
          model: optModel
          pred_cost: predicted costs
          lambd: Black-Box parameters for function smoothing
          processes: number of processors, 1 for single-core, 0 for number of CPUs
        """
        ins_num = len(pred_cost)
        # check model
        assert isinstance(model, optModel), 'arg model is not an optModel'
        # get device
        device = pred_cost.device
        # get num of processors
        processes = _SPO_FUNC_BB_PROCESSES
        # convert tenstor
        cp = pred_cost.to('cpu').numpy()
        # single-core
        if processes == 1:
            sol = []
            for i in range(ins_num):
                # solve
                model.setObj(cp[i])
                solp, _ = model.solve()
                sol.append(solp)
        # multi-core
        else:
            # number of processes
            processes = mp.cpu_count() if not processes else processes
            # get class
            model_name = type(model).__name__
            # get args
            args = getArgs(model)
            # parallel computing
            with ProcessingPool(processes) as pool:
                sol = pool.amap(solveWithObj4Par, cp, [args]*ins_num, [model_name]*ins_num).get()
        # convert to tensor
        pred_sol = torch.FloatTensor(sol).to(device)
        # save
        ctx.save_for_backward(pred_cost, pred_sol)
        ctx.model = model
        ctx.lambd = lambd
        return pred_sol

    @staticmethod
    def backward(ctx, grad_output):
        pred_cost, pred_sol = ctx.saved_tensors
        ins_num = len(pred_cost)
        # get device
        device = pred_cost.device
        # get num of processors
        processes = _SPO_FUNC_BB_PROCESSES
        # convert tenstor
        cp = pred_cost.to('cpu').numpy()
        wp = pred_sol.to('cpu').numpy()
        dl = grad_output.to('cpu').numpy()
        # perturbed costs
        cq = cp + ctx.lambd * dl
        # single-core
        if processes == 1:
            grad = []
            for i in range(len(cp)):
                # solve
                ctx.model.setObj(cq[i])
                solq, _ = ctx.model.solve()
                # gradient of continuous interpolation
                grad.append((solq - wp[i]) / ctx.lambd)
        # multi-core
        else:
            # get class
            model_name = type(ctx.model).__name__
            # get args
            args = getArgs(ctx.model)
            # number of processes
            processes = mp.cpu_count() if not processes else processes
            # parallel computing
            with ProcessingPool(processes) as pool:
                sol = pool.amap(solveWithObj4Par, cq, [args]*ins_num, [model_name]*ins_num).get()
            # get gradient
            grad = np.array(sol) - wp / ctx.lambd
        grad = torch.FloatTensor(grad).to(device)
        return None, grad, None
