#!/usr/bin/env python
"""
Knapsack problem
"""

from __future__ import annotations

import contextlib

with contextlib.suppress(ImportError):
    import gurobipy as gp
    from gurobipy import GRB

from pyepo.model.bases import knapsackBase
from pyepo.model.grb.grbmodel import optGrbModel


class knapsackModel(knapsackBase, optGrbModel):
    """
    Gurobi-backed knapsack.

    Attributes:
        _model (GurobiPy model): Gurobi model
        weights (np.ndarray): Weights of items
        capacity (np.ndarray): Total capacity
        items (list): List of item index
    """

    def _getModel(self) -> tuple:
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = gp.Model("knapsack")
        # variables
        x = m.addMVar(len(self.items), name="x", vtype=GRB.BINARY)
        # sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
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
    LP relaxation of the Gurobi knapsack.
    """

    def _getModel(self) -> tuple:
        """
        A method to build Gurobi model
        """
        # create a model
        m = gp.Model("knapsack")
        # variables
        x = m.addMVar(len(self.items), name="x", ub=1)
        # sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(self.weights @ x <= self.capacity)
        return m, x

    def relax(self) -> knapsackModelRel:
        """
        A forbidden method to relax MIP model
        """
        raise RuntimeError("Model has already been relaxed.")
