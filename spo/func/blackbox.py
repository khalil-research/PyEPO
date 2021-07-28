#!/usr/bin/env python
# coding: utf-8

import inspect
import torch
from torch.autograd import Function
import numpy as np
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
    @staticmethod
    def forward(ctx, model, pred_cost, lambd):
        ins_num = len(pred_cost)
        # check model
        assert isinstance(model, optModel), 'arg model is not an optModel'
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.to('cpu').numpy()
        # get class
        model_name = type(model).__name__
        # get args
        args = getArgs(model)
        # parallel computing
        with ProcessingPool() as pool:
            wp = pool.amap(solveWithObj4Par, cp, [args]*ins_num, [model_name]*ins_num).get()
        # convert to tensor
        pred_sol = torch.FloatTensor(wp).to(device)
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
        # convert tenstor
        cp = pred_cost.to('cpu').numpy()
        wp = pred_sol.to('cpu').numpy()
        dl = grad_output.to('cpu').numpy()
        # perturbed costs
        cq = cp + ctx.lambd * dl
        # get class
        model_name = type(ctx.model).__name__
        # get args
        args = getArgs(ctx.model)
        # parallel computing
        with ProcessingPool() as pool:
            sol = pool.amap(solveWithObj4Par, cq, [args]*ins_num, [model_name]*ins_num).get()
        # get gradient
        grad = []
        for i in range(ins_num):
            grad.append((sol[i] - wp[i]) / ctx.lambd)
        grad = torch.FloatTensor(grad).to(device)
        return None, grad, None
