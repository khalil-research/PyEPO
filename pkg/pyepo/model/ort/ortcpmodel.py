#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on Google OR-Tools CP-SAT
"""

from copy import copy

import numpy as np
import torch

try:
    from ortools.sat.python import cp_model
    _HAS_ORTOOLS = True
except ImportError:
    _HAS_ORTOOLS = False

from pyepo import EPO
from pyepo.model.opt import optModel


class optOrtCpModel(optModel):
    """
    This is an abstract class for an OR-Tools CP-SAT optimization model

    Attributes:
        _model (cp_model.CpModel): OR-Tools CP-SAT model
    """

    _OBJ_SCALE = 1_000_000

    def __init__(self):
        if not _HAS_ORTOOLS:
            raise ImportError("OR-Tools is not installed. Please install ortools to use this feature.")
        self._extra_constrs = []
        super().__init__()

    def __repr__(self):
        return "optOrtCpModel " + self.__class__.__name__

    def setObj(self, c):
        """
        A method to set the objective function

        Args:
            c (np.ndarray / list): cost of objective function
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector does not match number of cost variables.")
        # check if c is a PyTorch tensor
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        else:
            c = np.asarray(c, dtype=np.float64)
        # scale float to int
        scaled = (c * self._OBJ_SCALE).astype(np.int64)
        # set obj
        obj_expr = sum(int(scaled[i]) * self.x[k] for i, k in enumerate(self.x))
        if self.modelSense == EPO.MAXIMIZE:
            self._model.Maximize(obj_expr)
        else:
            self._model.Minimize(obj_expr)

    def solve(self):
        """
        A method to solve the model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        solver = cp_model.CpSolver()
        status = solver.Solve(self._model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                "Solver did not find a solution. Status: {}".format(
                    solver.StatusName(status)))
        sol = np.array([solver.Value(self.x[k]) for k in self.x], dtype=np.float32)
        obj = solver.ObjectiveValue() / self._OBJ_SCALE
        return sol, obj

    def copy(self):
        """
        A method to copy the model

        Returns:
            optModel: new copied model
        """
        new_model = copy(self)
        new_model._extra_constrs = list(self._extra_constrs)
        # rebuild model from scratch
        new_model._model, new_model.x = new_model._getModel()
        # replay extra constraints
        for coefs, rhs in new_model._extra_constrs:
            new_model._model.Add(
                sum(int(coefs[i]) * new_model.x[k]
                    for i, k in enumerate(new_model.x)) <= int(rhs))
        return new_model

    def addConstr(self, coefs, rhs):
        """
        A method to add a new constraint

        Args:
            coefs (np.ndarray / list): coefficients of new constraint
            rhs (float): right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        if len(coefs) != self.num_cost:
            raise ValueError("Size of coef vector does not match number of cost variables.")
        # scale to int
        scale = self._OBJ_SCALE
        scaled_coefs = [int(round(float(c) * scale)) for c in coefs]
        scaled_rhs = int(round(float(rhs) * scale))
        # copy
        new_model = self.copy()
        # store and add constraint
        new_model._extra_constrs.append((scaled_coefs, scaled_rhs))
        new_model._model.Add(
            sum(scaled_coefs[i] * new_model.x[k]
                for i, k in enumerate(new_model.x)) <= scaled_rhs)
        return new_model

    def relax(self):
        """
        CP-SAT does not support LP relaxation.
        """
        raise RuntimeError("CP-SAT does not support LP relaxation.")
