#!/usr/bin/env python
# coding: utf-8
"""
Knapsack problem
"""

from pyomo import environ as pe

from spo.model.omo import optOmoModel


class knapsackModel(optOmoModel):
    """
    This class is optimization model for knapsack problem

    Args:
        weights (ndarray): weights of items
        capacity (float): total capacity
        solver: optimization solver
    """

    def __init__(self, weights, capacity, solver="glpk"):
        self.weights = weights
        self.capacity = capacity
        self.items = list(range(len(self.weights)))
        super().__init__(solver)

    @property
    def num_cost(self):
        return len(self.items)

    def _getModel(self):
        """
        A method to build pyomo model
        """
        # ceate a model
        m = pe.ConcreteModel("knapsack")
        # parameters
        m.its = pe.Set(initialize=self.items)
        # varibles
        self.x = pe.Var(m.its, domain=pe.Binary)
        m.x = self.x
        # constraints
        m.cons = pe.ConstraintList()
        m.cons.add(sum(self.weights[i] * self.x[i]
                   for i in self.items) <= self.capacity)
        return m

    def relax(self):
        """
        A method to relax model
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
        # ceate a model
        m = pe.ConcreteModel("knapsack")
        # parameters
        m.its = pe.Set(initialize=self.items)
        # varibles
        self.x = pe.Var(m.its, domain=pe.PositiveReals, bounds=(0,1))
        m.x = self.x
        # constraints
        m.cons = pe.ConstraintList()
        m.cons.add(sum(self.weights[i] * self.x[i]
                   for i in self.items) <= self.capacity)
        return m

    def relax(self):
        """
        A forbidden method to relax MIP model
        """
        raise RuntimeError("Model has already been relaxed.")
