#!/usr/bin/env python
"""
Shortest path problem
"""

from __future__ import annotations

import contextlib
from collections import defaultdict

from pyepo.model.bases import shortestPathBase
from pyepo.model.omo.omomodel import optOmoModel

with contextlib.suppress(ImportError):
    from pyomo import environ as pe


class shortestPathModel(shortestPathBase, optOmoModel):
    """
    Pyomo-backed shortest path on a grid network.

    Attributes:
        _model (Pyomo model): Pyomo model
        solver (str): optimization solver in the background
        grid (tuple of int): Size of grid network
        arcs (list): List of arcs
    """

    def __init__(self, grid: tuple[int, int], solver: str = "glpk") -> None:
        """
        Args:
            grid: size of grid network
            solver: optimization solver in the background
        """
        super().__init__(grid, solver)

    def _getModel(self) -> tuple:
        """
        A method to build Pyomo model
        """
        # create a model
        m = pe.ConcreteModel("shortest path")
        # parameters
        m.arcs = pe.Set(initialize=self.arcs)
        # variables
        x = pe.Var(m.arcs, domain=pe.NonNegativeReals, bounds=(0, 1))
        m.x = x
        # build adjacency lists
        out_arcs = defaultdict(list)
        in_arcs = defaultdict(list)
        for e in self.arcs:
            out_arcs[e[0]].append(e)
            in_arcs[e[1]].append(e)
        # constraints
        m.cons = pe.ConstraintList()
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                expr = sum(x[e] for e in in_arcs[v]) - sum(x[e] for e in out_arcs[v])
                # source
                if i == 0 and j == 0:
                    m.cons.add(expr == -1)
                # sink
                elif i == self.grid[0] - 1 and j == self.grid[1] - 1:
                    m.cons.add(expr == 1)
                # transition
                else:
                    m.cons.add(expr == 0)
        return m, x
