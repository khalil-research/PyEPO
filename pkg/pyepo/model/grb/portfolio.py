#!/usr/bin/env python
"""
Portfolio problem
"""

from __future__ import annotations

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    pass

from pyepo.model.bases import portfolioBase
from pyepo.model.grb.grbmodel import optGrbModel


class portfolioModel(portfolioBase, optGrbModel):
    """
    Gurobi-backed Markowitz portfolio.

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_assets (int): number of assets
        covariance (numpy.ndarray): covariance matrix of the returns
        risk_level (float): risk level
    """

    def _getModel(self) -> tuple:
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = gp.Model("portfolio")
        # variables
        x = m.addMVar(self.num_assets, ub=1, name="x")
        # sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(x.sum() == 1, "budget")
        m.addConstr(x.T @ self.covariance @ x <= self.risk_level, "risk_limit")
        return m, x
