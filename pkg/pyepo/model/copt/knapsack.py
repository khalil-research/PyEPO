#!/usr/bin/env python
"""
Knapsack problem
"""

from __future__ import annotations

try:
    from coptpy import COPT
except ImportError:
    pass

from pyepo.model.bases import knapsackBase
from pyepo.model.copt.coptmodel import _get_envr, optCoptModel


class knapsackModel(knapsackBase, optCoptModel):
    """
    COPT-backed knapsack.

    Attributes:
        _model (COPT model): COPT model
        weights (np.ndarray): weights of items
        capacity (np.ndarray): total capacity
        items (list): list of item index
    """

    def _getModel(self) -> tuple:
        """
        A method to build COPT model
        """
        m = _get_envr().createModel("knapsack")
        num_items = self.weights.shape[1]
        x = m.addMVar(num_items, vtype=COPT.BINARY, nameprefix="x")
        m.setObjSense(COPT.MAXIMIZE)
        m.addConstr(self.weights @ x <= self.capacity)
        return m, x

    def relax(self) -> knapsackModelRel:
        """
        A method to get linear relaxation model
        """
        # copy
        model_rel = knapsackModelRel(self.weights, self.capacity)
        # replay user cuts on the relaxation
        for coefs, rhs in self._extra_constrs:
            model_rel = model_rel.addConstr(coefs, rhs)
        return model_rel


class knapsackModelRel(knapsackModel):
    """
    LP relaxation of the COPT knapsack.
    """

    def _getModel(self) -> tuple:
        """
        A method to build COPT model
        """
        m = _get_envr().createModel("knapsack")
        num_items = self.weights.shape[1]
        x = m.addMVar(num_items, lb=0.0, ub=1.0, vtype=COPT.CONTINUOUS, nameprefix="x")
        m.setObjSense(COPT.MAXIMIZE)
        m.addConstr(self.weights @ x <= self.capacity)
        return m, x

    def relax(self) -> knapsackModelRel:
        """
        A forbidden method to relax MIP model
        """
        raise RuntimeError("Model has already been relaxed.")
