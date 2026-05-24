#!/usr/bin/env python
"""
Portfolio problem
"""

from __future__ import annotations

from coptpy import COPT, Envr

from pyepo.model.bases import portfolioBase
from pyepo.model.copt.coptmodel import optCoptModel


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
        m = Envr().createModel("portfolio")
        x = m.addMVar(self.num_assets, lb=0.0, ub=1.0, vtype=COPT.CONTINUOUS, nameprefix="x")
        m.setObjSense(COPT.MAXIMIZE)
        m.addConstr(x.sum() == 1, "budget")
        m.addQConstr(x @ self.covariance @ x <= self.risk_level, "risk_limit")
        return m, x


if __name__ == "__main__":
    import random

    from pyepo.data.portfolio import genData

    # random seed
    random.seed(42)
    # set random cost for test
    covariance, _, revenue = genData(num_data=100, num_features=4, num_assets=50, deg=2)

    # solve model
    optmodel = portfolioModel(num_assets=50, covariance=covariance)  # init model
    optmodel = optmodel.copy()
    optmodel.setObj(revenue[0])  # set objective function
    sol, obj = optmodel.solve()  # solve
    # print res
    print(f"Obj: {obj}")
    for i in range(50):
        if sol[i] > 1e-3:
            print(f"Asset {i}: {100 * sol[i]:.2f}%")

    # add constraint
    optmodel = optmodel.addConstr([1] * 50, 30)
    optmodel.setObj(revenue[0])  # set objective function
    sol, obj = optmodel.solve()  # solve
    # print res
    print(f"Obj: {obj}")
    for i in range(50):
        if sol[i] > 1e-3:
            print(f"Asset {i}: {100 * sol[i]:.2f}%")
