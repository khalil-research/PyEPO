#!/usr/bin/env python
# coding: utf-8
"""
Knapsack problem
"""

import numpy as np
from pyomo import environ as pe

from pyepo import EPO
from pyepo.model.omo.omomodel import optOmoModel


class knapsackModel(optOmoModel):
    """
    This class is optimization model for knapsack problem

    Attributes:
        _model (PyOmo model): Pyomo model
        solver (str): optimization solver in the background
        weights (np.ndarray): weights of items
        capacity (np.ndarray): total capacity
        items (list): list of item index
    """

    def __init__(self, weights, capacity, solver="glpk"):
        """
        Args:
            weights (np.ndarray / list): weights of items
            capacity (np.ndarray / list): total capacity
            solver (str): optimization solver in the background
        """
        self.weights = np.array(weights)
        self.capacity = np.array(capacity)
        self.items = list(range(self.weights.shape[1]))
        super().__init__(solver)

    def _getModel(self):
        """
        A method to build pyomo model
        """
        # sense
        self.modelSense = EPO.MAXIMIZE
        # ceate a model
        m = pe.ConcreteModel("knapsack")
        # parameters
        m.its = pe.Set(initialize=self.items)
        # varibles
        x = pe.Var(m.its, domain=pe.Binary)
        m.x = x
        # constraints
        m.cons = pe.ConstraintList()
        for i in range(len(self.capacity)):
            m.cons.add(sum(self.weights[i,j] * x[j]
                       for j in self.items) <= self.capacity[i])

        return m, x

    def relax(self):
        """
        A method to get linear relaxation model
        """
        # copy
        model_rel = knapsackModelRel(self.weights, self.capacity)
        return model_rel


class knapsackModelRel(knapsackModel):
    """
    This class is relaxed optimization model for knapsack problem.
    """

    def _getModel(self):
        """
        A method to build pyomo
        """
        # sense
        self.modelSense = EPO.MAXIMIZE
        # ceate a model
        m = pe.ConcreteModel("knapsack")
        # parameters
        m.its = pe.Set(initialize=self.items)
        # varibles
        x = pe.Var(m.its, domain=pe.PositiveReals, bounds=(0,1))
        m.x = x
        # constraints
        m.cons = pe.ConstraintList()
        for i in range(len(self.capacity)):
            m.cons.add(sum(self.weights[i,j] * x[j]
                       for j in self.items) <= self.capacity[i])
        return m, x

    def relax(self):
        """
        A forbidden method to relax MIP model
        """
        raise RuntimeError("Model has already been relaxed.")
