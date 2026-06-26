#!/usr/bin/env python
"""
Portfolio problem
"""

from __future__ import annotations

try:
    from coptpy import COPT
except ImportError:
    COPT = None

from pyepo.model.bases import portfolioBase
from pyepo.model.copt.coptmodel import _get_envr, optCoptModel


class portfolioModel(portfolioBase, optCoptModel):
    """
    COPT-backed Markowitz portfolio.

    Attributes:
        _model (COPT model): COPT model
        num_assets (int): number of assets
        covariance (numpy.ndarray): covariance matrix of the returns
        risk_level (float): risk level
    """

    def _getModel(self) -> tuple:
        """
        A method to build COPT model

        Returns:
            tuple: optimization model and variables
        """
        m = _get_envr().createModel("portfolio")
        x = m.addMVar(self.num_assets, lb=0.0, ub=1.0, vtype=COPT.CONTINUOUS, nameprefix="x")
        m.setObjSense(COPT.MAXIMIZE)
        m.addConstr(x.sum() == 1, "budget")
        m.addQConstr(x @ self.covariance @ x <= self.risk_level, "risk_limit")
        return m, x
