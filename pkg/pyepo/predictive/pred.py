#!/usr/bin/env python
# coding: utf-8
"""
Abstract predictive prescription model
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
from pyepo.model.opt import optModel

class PredictivePrescription(ABC):
    """
    This is an abstract class for predicitive prescription model
    """
    def __init__(self, model):
        self.model: optModel = model
        pass
    
    @abstractmethod
    def _get_weights(self, x):
        """
        An abstract method to gather the weights for the prediction
        """
        raise NotImplementedError

    def optimize(self, x): 
        # Predict
        with torch.no_grad():
            weights = self._get_weights(x)

        if isinstance(weights, torch.Tensor):
            if weights.is_cuda:
                weights = weights.detach().cpu()
            else:
                weights = weights.detach()

            if weights.dim() == 2 and weights.size(0) == 1:
                weights = weights.squeeze(0)

            weights = weights.numpy()
                    
        # Optimize
        self.model.setWeightObj(weights, self.costs)
        sol, obj = self.model.solve()

        if isinstance(sol, torch.Tensor):
            sol = sol.detach().cpu().numpy()

        return sol, obj