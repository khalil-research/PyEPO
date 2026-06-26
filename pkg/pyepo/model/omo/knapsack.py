#!/usr/bin/env python
"""
Knapsack problem
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyepo.model.bases import knapsackBase
from pyepo.model.omo.omomodel import optOmoModel

try:
    from pyomo import environ as pe
except ImportError:
    pass

if TYPE_CHECKING:
    import numpy as np


class knapsackModel(knapsackBase, optOmoModel):
    """
    Pyomo-backed knapsack.

    Attributes:
        _model (Pyomo model): Pyomo model
        solver (str): optimization solver in the background
        weights (np.ndarray): weights of items
        capacity (np.ndarray): total capacity
        items (list): list of item index
    """

    def __init__(
        self,
        weights: np.ndarray | list,
        capacity: np.ndarray | list,
        solver: str = "glpk",
    ) -> None:
        """
        Args:
            weights: weights of items
            capacity: total capacity
            solver: optimization solver in the background
        """
        super().__init__(weights, capacity, solver)

    def _getModel(self) -> tuple:
        """
        A method to build Pyomo model
        """
        # create a model
        m = pe.ConcreteModel("knapsack")
        # parameters
        m.its = pe.Set(initialize=self.items)
        # variables
        x = pe.Var(m.its, domain=pe.Binary)
        m.x = x
        # constraints
        m.cons = pe.ConstraintList()
        for i in range(len(self.capacity)):
            m.cons.add(sum(self.weights[i, j] * x[j] for j in self.items) <= self.capacity[i])
        return m, x

    def relax(self) -> knapsackModelRel:
        """
        A method to get linear relaxation model
        """
        model_rel = knapsackModelRel(self.weights, self.capacity, self.solver)
        # replay user cuts on the relaxation
        for coefs, rhs in self._extra_constrs:
            model_rel = model_rel.addConstr(coefs, rhs)
        return model_rel


class knapsackModelRel(knapsackModel):
    """
    LP relaxation of the Pyomo knapsack.
    """

    def _getModel(self) -> tuple:
        """
        A method to build Pyomo model
        """
        # create a model
        m = pe.ConcreteModel("knapsack")
        # parameters
        m.its = pe.Set(initialize=self.items)
        # variables
        x = pe.Var(m.its, domain=pe.NonNegativeReals, bounds=(0, 1))
        m.x = x
        # constraints
        m.cons = pe.ConstraintList()
        for i in range(len(self.capacity)):
            m.cons.add(sum(self.weights[i, j] * x[j] for j in self.items) <= self.capacity[i])
        return m, x

    def relax(self) -> knapsackModelRel:
        """
        A forbidden method to relax MIP model
        """
        raise RuntimeError("Model has already been relaxed.")
