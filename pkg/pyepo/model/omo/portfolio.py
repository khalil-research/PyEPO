#!/usr/bin/env python
# coding: utf-8
"""
Portfolio problem
"""

import numpy as np

from pyepo import EPO
from pyepo.model.omo.omomodel import optOmoModel

try:
    from pyomo import environ as pe
    _HAS_PYOMO = True
except ImportError:
    _HAS_PYOMO = False


class portfolioModel(optOmoModel):
    """
    This class is an optimization model for the portfolio problem

    Attributes:
        _model (Pyomo model): Pyomo model
        solver (str): optimization solver in the background
        num_assets (int): number of assets
        covariance (numpy.ndarray): covariance matrix of the returns
        risk_level (float): risk level
    """

    def __init__(self, num_assets, covariance, gamma=2.25, solver="glpk"):
        """
        Args:
            num_assets (int): number of assets
            covariance (numpy.ndarray): covariance matrix of the returns
            gamma (float): risk level parameter
            solver (str): optimization solver in the background
        """
        self.num_assets = num_assets
        self.covariance = covariance
        self.risk_level = self._getRiskLevel(gamma)
        super().__init__(solver)

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
        A method to build Pyomo model

        Returns:
            tuple: optimization model and variables
        """
        # sense
        self.modelSense = EPO.MAXIMIZE
        # create a model
        m = pe.ConcreteModel("portfolio")
        # parameters
        m.assets = pe.Set(initialize=range(self.num_assets))
        # variables
        x = pe.Var(m.assets, domain=pe.NonNegativeReals, bounds=(0, 1))
        m.x = x
        # constraints
        m.cons = pe.ConstraintList()
        m.cons.add(sum(x[i] for i in range(self.num_assets)) == 1)
        m.cons.add(sum(self.covariance[i,j] * x[i] * x[j]
                   for i in range(self.num_assets)
                   for j in range(self.num_assets)) <= self.risk_level)
        return m, x


if __name__ == "__main__":

    import random
    from pyepo.data.portfolio import genData
    # random seed
    random.seed(42)
    # set random cost for test
    covariance, _, revenue = genData(num_data=100, num_features=4, num_assets=50, deg=2)

    # solve model
    optmodel = portfolioModel(num_assets=50, covariance=covariance, solver="gurobi") # init model
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
