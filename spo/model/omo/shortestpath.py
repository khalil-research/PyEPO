#!/usr/bin/env python
# coding: utf-8
"""
Shortest path problem
"""

from pyomo import environ as pe

from spo.model.omo import optOmoModel


class shortestPathModel(optOmoModel):
    """
    This class is optimization model for shortest path problem

    Attributes:
        _model (PyOmo model): Pyomo model
        solver (str): optimization solver in the background
        grid (tuple): size of grid network
        arcs (list): list of arcs
    """

    def __init__(self, grid, solver="glpk"):
        """
        Args:
            grid (tuple): size of grid network
            solver (str): optimization solver in the background
        """
        self.grid = grid
        self.arcs = self._getArcs()
        super().__init__(solver)

    def _getArcs(self):
        """
        A method to get list of arcs for grid network

        Returns:
            list: arcs
        """
        arcs = []
        for i in range(self.grid[0]):
            # edges on rows
            for j in range(self.grid[1] - 1):
                v = i * self.grid[1] + j
                arcs.append((v, v + 1))
            # edges in columns
            if i == self.grid[0] - 1:
                continue
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                arcs.append((v, v + self.grid[1]))
        return arcs

    @property
    def num_cost(self):
        return len(self.arcs)

    def _getModel(self):
        """
        A method to build pyomo model
        """
        # ceate a model
        m = pe.ConcreteModel("shortest path")
        # parameters
        m.arcs = pe.Set(initialize=self.arcs)
        # varibles
        self.x = pe.Var(m.arcs, domain=pe.PositiveReals, bounds=(0,1))
        m.x = self.x
        # constraints
        m.cons = pe.ConstraintList()
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                v = v = i * self.grid[1] + j
                expr = 0
                for e in self.arcs:
                    # flow in
                    if v == e[1]:
                        expr += self.x[e]
                    # flow out
                    elif v == e[0]:
                        expr -= self.x[e]
                # source
                if i == 0 and j == 0:
                    m.cons.add(expr == -1)
                # sink
                elif i == self.grid[0] - 1 and j == self.grid[0] - 1:
                    m.cons.add(expr == 1)
                # transition
                else:
                    m.cons.add(expr == 0)
        return m
