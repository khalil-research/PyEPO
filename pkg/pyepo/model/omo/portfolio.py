#!/usr/bin/env python
"""
Portfolio problem
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyepo.model.bases import portfolioBase
from pyepo.model.omo.omomodel import optOmoModel

try:
    from pyomo import environ as pe
except ImportError:
    pass

if TYPE_CHECKING:
    import numpy as np


class portfolioModel(portfolioBase, optOmoModel):
    """
    Pyomo-backed Markowitz portfolio.

    Attributes:
        _model (Pyomo model): Pyomo model
        solver (str): optimization solver in the background
        num_assets (int): number of assets
        covariance (numpy.ndarray): covariance matrix of the returns
        risk_level (float): risk level
    """

    def __init__(
        self,
        num_assets: int,
        covariance: np.ndarray,
        gamma: float = 2.25,
        solver: str = "glpk",
    ) -> None:
        """
        Args:
            num_assets: number of assets
            covariance: covariance matrix of the returns
            gamma: risk level parameter
            solver: optimization solver in the background
        """
        super().__init__(num_assets, covariance, gamma, solver)

    def _getModel(self) -> tuple:
        """
        A method to build Pyomo model

        Returns:
            tuple: optimization model and variables
        """
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
        m.cons.add(
            sum(
                self.covariance[i, j] * x[i] * x[j]
                for i in range(self.num_assets)
                for j in range(self.num_assets)
            )
            <= self.risk_level
        )
        return m, x


if __name__ == "__main__":
    import random

    from pyepo.data.portfolio import genData

    # random seed
    random.seed(42)
    # set random cost for test
    covariance, _, revenue = genData(num_data=100, num_features=4, num_assets=50, deg=2)

    # solve model
    optmodel = portfolioModel(num_assets=50, covariance=covariance, solver="gurobi")  # init model
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
