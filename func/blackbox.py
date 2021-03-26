#!/usr/bin/env python
# coding: utf-8

import torch
from torch.autograd import Function
import numpy as np
from model import optModel

class blackboxOpt(Function):
    """
    block-box optimizer function, which is diffenretiable to introduce blocks into neural networks.
    """
    @staticmethod
    def forward(ctx, model, pred_cost, lambd):
        # check model
        assert isinstance(model, optModel), 'arg model is not an optModel'
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.to('cpu').numpy()
        # predicted solutions
        wp = []
        for i in range(len(cp)):
            # solve
            model.setObj(cp[i])
            solp, _ = model.solve()
            wp.append(solp)
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
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.to('cpu').numpy()
        wp = pred_sol.to('cpu').numpy()
        dl = grad_output.to('cpu').numpy()
        # perturbed costs
        cq = cp + ctx.lambd * dl
        # init gradient
        grad = []
        for i in range(len(cp)):
            # solve
            ctx.model.setObj(cq[i])
            solq, _ = ctx.model.solve()
            # gradient of continuous interpolation
            grad.append((solq - wp[i]) / ctx.lambd)
        # convert to tensor
        grad = torch.FloatTensor(grad).to(device)
        return None, grad, None
