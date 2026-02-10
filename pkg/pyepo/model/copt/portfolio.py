#!/usr/bin/env python
# coding: utf-8
"""
Portfolio problem
"""

import numpy as np
from coptpy import Envr
from coptpy import COPT

from pyepo import EPO
from pyepo.model.copt.coptmodel import optCoptModel


class portfolioModel(optCoptModel):
    """
    This class is an optimization model for the portfolio problem

    Attributes:
        _model (COPT model): COPT model
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

        Args:
            gamma (float): risk level parameter

        Returns:
            float: risk level
        """
        risk_level = gamma * np.mean(self.covariance)
        return risk_level

    def _getModel(self):
        """
        A method to build COPT model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = Envr().createModel("portfolio")
        # variables
        x = m.addVars(range(self.num_assets), nameprefix='x',
                       vtype=COPT.CONTINUOUS, lb=0, ub=1)
        # sense
        m.setObjSense(COPT.MAXIMIZE)
        # constraints
        m.addConstr(sum(x[i] for i in range(self.num_assets)) == 1, "budget")
        m.addQConstr(sum(self.covariance[i,j] * x[i] * x[j]
                     for i in range(self.num_assets)
                     for j in range(self.num_assets)) <= self.risk_level,
                     "risk_limit")
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
