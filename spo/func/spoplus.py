#!/usr/bin/env python
# coding: utf-8

import inspect
import torch
from torch.autograd import Function
import numpy as np
from pathos.multiprocessing import ProcessingPool
import spo
from spo.model import optModel

def solveWithObj4Par(obj, args, model_name):
    """
    global solve function for parallel
    """
    # build model
    model = eval('spo.model.{}'.format(model_name))(**args)
    # set obj
    model.setObj(obj)
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

    @staticmethod
    def forward(ctx, model, pred_cost, true_cost, true_sol, true_obj):
        # check model
        assert isinstance(model, optModel), 'arg model is not an optModel'
        # get device
        device = pred_cost.device
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
        # get class
        model_name = type(model).__name__
        # get args
        args = getArgs(model)
        # parallel computing
        with ProcessingPool() as pool:
            res = pool.amap(solveWithObj4Par, 2*cp-c, [args]*ins_num, [model_name]*ins_num).get()
        # get res
        sol = np.array(list(map(lambda x: x[0], res)))
        obj = np.array(list(map(lambda x: x[1], res)))
        # calculate loss
        loss = []
        for i in range(ins_num):
            loss.append(- obj[i] + 2 * np.dot(cp[i], w[i]) - z[i])
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
        return None, grad_output * grad, None, None, None
