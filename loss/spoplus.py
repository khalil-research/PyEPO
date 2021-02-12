#!/usr/bin/env python
# coding: utf-8

from torch.autograd import Function
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
    def __init__(self, model):
        """
        Args:
            model: instance of optModel
        """
        super().__init__()
        assert isinstance(model, optModel), 'arg model is not an optModel'
        self.m = model

    def forward(pred_cost, true_cost, true_sol):
        pass

    def backward(grad_output):
        pass
