#!/usr/bin/env python
"""
Shortest path problem
"""

from __future__ import annotations

from collections import defaultdict

try:
    from ortools.linear_solver import pywraplp
    from ortools.sat.python import cp_model
except ImportError:
    pass

from pyepo.model.bases import shortestPathBase
from pyepo.model.ort.ortcpmodel import optOrtCpModel
from pyepo.model.ort.ortmodel import optOrtModel


class shortestPathModel(shortestPathBase, optOrtModel):
    """
    OR-Tools (pywraplp) backed shortest path on a grid network.

    Attributes:
        _model (pywraplp.Solver): OR-Tools linear solver
        solver (str): solver backend
        grid (tuple of int): Size of grid network
        arcs (list): List of arcs
    """

    def __init__(self, grid: tuple[int, int], solver: str = "glop") -> None:
        """
        Args:
            grid: size of grid network
            solver: solver backend for pywraplp
        """
        super().__init__(grid, solver)

    def _getModel(self) -> tuple:
        """
        A method to build OR-Tools pywraplp model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = pywraplp.Solver.CreateSolver(self.solver.upper())
        if m is None:
            raise RuntimeError(f"Solver '{self.solver}' is not available in OR-Tools.")
        # variables
        x = {e: m.NumVar(0, 1, f"x_{e}") for e in self.arcs}
        # build adjacency lists
        out_arcs = defaultdict(list)
        in_arcs = defaultdict(list)
        for e in self.arcs:
            out_arcs[e[0]].append(e)
            in_arcs[e[1]].append(e)
        # constraints
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                expr = sum(x[e] for e in in_arcs[v]) - sum(x[e] for e in out_arcs[v])
                # source
                if i == 0 and j == 0:
                    m.Add(expr == -1)
                # sink
                elif i == self.grid[0] - 1 and j == self.grid[1] - 1:
                    m.Add(expr == 1)
                # transition
                else:
                    m.Add(expr == 0)
        return m, x


class shortestPathCpModel(shortestPathBase, optOrtCpModel):
    """
    OR-Tools CP-SAT backed shortest path on a grid network.

    Attributes:
        _model (cp_model.CpModel): OR-Tools CP-SAT model
        grid (tuple of int): Size of grid network
        arcs (list): List of arcs
    """

    def _getModel(self) -> tuple:
        """
        A method to build OR-Tools CP-SAT model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = cp_model.CpModel()
        # variables (boolean: 0 or 1 flow)
        x = {e: m.NewBoolVar(f"x_{e}") for e in self.arcs}
        # build adjacency lists
        out_arcs = defaultdict(list)
        in_arcs = defaultdict(list)
        for e in self.arcs:
            out_arcs[e[0]].append(e)
            in_arcs[e[1]].append(e)
        # constraints
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                expr = sum(x[e] for e in in_arcs[v]) - sum(x[e] for e in out_arcs[v])
                # source
                if i == 0 and j == 0:
                    m.Add(expr == -1)
                # sink
                elif i == self.grid[0] - 1 and j == self.grid[1] - 1:
                    m.Add(expr == 1)
                # transition
                else:
                    m.Add(expr == 0)
        return m, x


if __name__ == "__main__":
    import random

    # random seed
    random.seed(42)
    # set random cost for test
    cost = [random.random() for _ in range(40)]

    # ---- pywraplp ----
    print("=== pywraplp (GLOP) ===")
    optmodel = shortestPathModel(grid=(5, 5))
    optmodel = optmodel.copy()
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print(f"Obj: {obj}")
    for i, e in enumerate(optmodel.arcs):
        if sol[i] > 1e-3:
            print(e)

    # add constraint
    print("\n=== pywraplp + addConstr ===")
    optmodel2 = optmodel.addConstr([1] * 40, 30)
    optmodel2.setObj(cost)
    sol, obj = optmodel2.solve()
    print(f"Obj: {obj}")
    for i, e in enumerate(optmodel.arcs):
        if sol[i] > 1e-3:
            print(e)

    # ---- CP-SAT ----
    print("\n=== CP-SAT ===")
    optmodel_cp = shortestPathCpModel(grid=(5, 5))
    optmodel_cp = optmodel_cp.copy()
    optmodel_cp.setObj(cost)
    sol, obj = optmodel_cp.solve()
    print(f"Obj: {obj}")
    for i, e in enumerate(optmodel_cp.arcs):
        if sol[i] > 1e-3:
            print(e)
