#!/usr/bin/env python
# coding: utf-8
"""
Portfolio problem
"""

import numpy as np
try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

from pyepo.model.grb.grbmodel import optGrbModel


class portfolioModel(optGrbModel):
    """
    This class is an optimization model for portfolio problem

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_assets (int): number of assets
        covariance (numpy.ndarray): covariance matrix of the returns
        risk_level (float): risk level
    """

    def __init__(self, num_assets, covariance, gamma=2.25):
        """
        Args:
            num_assets (int): number of assets
            covariance (numpy.ndarray): covariance matrix of the returns
            gamma (float): risk level parameter
        """
        self.num_assets = num_assets
        self.covariance = covariance
        self.risk_level = self._getRiskLevel(gamma)
        super().__init__()

    def _getRiskLevel(self, gamma):
        """
        A method to calculate the risk level

        Returns:
            float: risk level
        """
        risk_level = gamma * np.mean(self.covariance)
        return risk_level

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("portfolio")
        # varibles
        x = m.addMVar(self.num_assets, ub=1, name="x")
        # sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(x.sum() == 1, "budget")
        m.addConstr(x.T @ self.covariance @ x <= self.risk_level, "risk_limit")
        return m, x


if __name__ == "__main__":

    import random
    from pyepo.data.portfolio import genData
    # random seed
    random.seed(42)
    # set random cost for test
    covariance, _, revenue = genData(num_data=100, num_features=4, num_assets=50, deg=2)

    # solve model
    optmodel = portfolioModel(num_assets=50, covariance=covariance) # init model
    optmodel = optmodel.copy()
    optmodel.setObj(revenue[0]) # set objective function
    sol, obj = optmodel.solve() # solve
    # print res
    print('Obj: {}'.format(obj))
    for i in range(50):
        if sol[i] > 1e-3:
            print("Asset {}: {:.2f}%".format(i, 100*sol[i]))

    # add constraint
    optmodel = optmodel.addConstr([1]*50, 30)
    optmodel.setObj(revenue[0]) # set objective function
    sol, obj = optmodel.solve() # solve
    # print res
    print('Obj: {}'.format(obj))
    for i in range(50):
        if sol[i] > 1e-3:
            print("Asset {}: {:.2f}%".format(i, 100*sol[i]))
