#!/usr/bin/env python
# coding: utf-8

import torch
from torch.autograd import Function
import numpy as np
from model import optModel

class SPOPlusLoss(Function):
    """
    SPO+ Loss function, a surrogate loss function of SPO Loss, which
    measure the decision error (optimality gap) of optimization problem.

    For SPO/SPO+ Loss, the objective function is linear and constraints
    are known and fixed, but the cost vector need to be predicted from
    contextual data.

    The SPO+ Loss is convex with subgradient. Thus, allows us to design
    an algorithm based on stochastic gradient descent.
    """
    def __init__(self):
        """
        Args:
            model: instance of optModel
        """
        super().__init__()

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
        # init loss
        loss = []
        wq = []
        for i in range(len(z)):
            assert abs(z[i] - np.dot(c[i], w[i])) < 1e-3, 'Solution {} does not macth the objective value {}.'.format(np.dot(c[i], w[i]), z[i])
            # solve
            model.setObj(2 * cp[i] - c[i])
            solq, objq = model.solve()
            # calculate loss
            loss.append(- objq + 2 * np.dot(cp[i], w[i]) - z[i])
            # solution
            wq.append(solq)
        # convert to tensor
        loss = torch.FloatTensor(loss).to(device)
        wq = torch.FloatTensor(wq).to(device)
        # save solutions
        ctx.save_for_backward(true_sol, wq)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        w, wq = ctx.saved_tensors
        grad = 2 * (w - wq).mean(0)
        return None, grad_output * grad, None, None, None
