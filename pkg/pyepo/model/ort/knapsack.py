#!/usr/bin/env python
# coding: utf-8
"""
Knapsack problem
"""

import numpy as np

try:
    from ortools.linear_solver import pywraplp
    from ortools.sat.python import cp_model
    _HAS_ORTOOLS = True
except ImportError:
    _HAS_ORTOOLS = False

from pyepo import EPO
from pyepo.model.ort.ortmodel import optOrtModel
from pyepo.model.ort.ortcpmodel import optOrtCpModel


# ============================================================
# pywraplp
# ============================================================

class knapsackModel(optOrtModel):
    """
    This class is an optimization model for the knapsack problem

    Attributes:
        _model (pywraplp.Solver): OR-Tools linear solver
        solver (str): solver backend
        weights (np.ndarray): weights of items
        capacity (np.ndarray): total capacity
        items (list): list of item index
    """

    def __init__(self, weights, capacity, solver="scip"):
        """
        Args:
            weights (np.ndarray / list): weights of items
            capacity (np.ndarray / list): total capacity
            solver (str): solver backend for pywraplp
        """
        self.weights = np.asarray(weights)
        self.capacity = np.asarray(capacity)
        self.items = list(range(self.weights.shape[1]))
        super().__init__(solver)

    def _getModel(self):
        """
        A method to build OR-Tools pywraplp model

        Returns:
            tuple: optimization model and variables
        """
        # sense
        self.modelSense = EPO.MAXIMIZE
        # create a model
        m = pywraplp.Solver.CreateSolver(self.solver.upper())
        if m is None:
            raise RuntimeError(
                "Solver '{}' is not available in OR-Tools.".format(self.solver))
        # variables
        x = {i: m.BoolVar("x_{}".format(i)) for i in self.items}
        # constraints
        for i in range(len(self.capacity)):
            m.Add(sum(float(self.weights[i, j]) * x[j]
                      for j in self.items) <= float(self.capacity[i]))
        return m, x

    def relax(self):
        """
        A method to get linear relaxation model
        """
        return knapsackModelRel(self.weights, self.capacity, self.solver)


class knapsackModelRel(knapsackModel):
    """
    This class is relaxed optimization model for knapsack problem.
    """

    def _getModel(self):
        """
        A method to build OR-Tools pywraplp model (relaxed)

        Returns:
            tuple: optimization model and variables
        """
        # sense
        self.modelSense = EPO.MAXIMIZE
        # create a model
        m = pywraplp.Solver.CreateSolver(self.solver.upper())
        if m is None:
            raise RuntimeError(
                "Solver '{}' is not available in OR-Tools.".format(self.solver))
        # variables (continuous)
        x = {i: m.NumVar(0, 1, "x_{}".format(i)) for i in self.items}
        # constraints
        for i in range(len(self.capacity)):
            m.Add(sum(float(self.weights[i, j]) * x[j]
                      for j in self.items) <= float(self.capacity[i]))
        return m, x

    def relax(self):
        """
        A forbidden method to relax MIP model
        """
        raise RuntimeError("Model has already been relaxed.")


# ============================================================
# CP-SAT
# ============================================================

class knapsackCpModel(optOrtCpModel):
    """
    This class is an optimization model for the knapsack problem using CP-SAT

    Attributes:
        _model (cp_model.CpModel): OR-Tools CP-SAT model
        weights (np.ndarray): weights of items
        capacity (np.ndarray): total capacity
        items (list): list of item index
    """

    def __init__(self, weights, capacity):
        """
        Args:
            weights (np.ndarray / list): weights of items
            capacity (np.ndarray / list): total capacity
        """
        self.weights = np.asarray(weights)
        self.capacity = np.asarray(capacity)
        self.items = list(range(self.weights.shape[1]))
        super().__init__()

    def _getModel(self):
        """
        A method to build OR-Tools CP-SAT model

        Returns:
            tuple: optimization model and variables
        """
        # sense
        self.modelSense = EPO.MAXIMIZE
        # create a model
        m = cp_model.CpModel()
        # variables
        x = {i: m.NewBoolVar("x_{}".format(i)) for i in self.items}
        # constraints (integer coefficients)
        for i in range(len(self.capacity)):
            m.Add(sum(int(self.weights[i, j]) * x[j]
                      for j in self.items) <= int(self.capacity[i]))
        return m, x


if __name__ == "__main__":

    # random seed
    np.random.seed(42)
    # set random cost for test
    cost = np.random.random(16)
    weights = np.random.choice(range(300, 800), size=(2, 16)) / 100
    capacity = [20, 20]

    # ---- pywraplp ----
    print("=== pywraplp (SCIP) ===")
    optmodel = knapsackModel(weights=weights, capacity=capacity)
    optmodel = optmodel.copy()
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print("Obj: {}".format(obj))
    for i in range(16):
        if sol[i] > 1e-3:
            print(i)

    # relax
    print("\n=== pywraplp relaxed ===")
    optmodel_rel = optmodel.relax()
    optmodel_rel.setObj(cost)
    sol, obj = optmodel_rel.solve()
    print("Obj: {}".format(obj))
    for i in range(16):
        if sol[i] > 1e-3:
            print("{}: {:.4f}".format(i, sol[i]))

    # add constraint
    print("\n=== pywraplp + addConstr ===")
    optmodel2 = optmodel.addConstr([1] * 16, 5)
    optmodel2.setObj(cost)
    sol, obj = optmodel2.solve()
    print("Obj: {}".format(obj))
    for i in range(16):
        if sol[i] > 1e-3:
            print(i)

    # ---- CP-SAT ----
    print("\n=== CP-SAT ===")
    # CP-SAT needs integer weights
    weights_int = np.round(weights).astype(int)
    capacity_int = [20, 20]
    optmodel_cp = knapsackCpModel(weights=weights_int, capacity=capacity_int)
    optmodel_cp = optmodel_cp.copy()
    optmodel_cp.setObj(cost)
    sol, obj = optmodel_cp.solve()
    print("Obj: {}".format(obj))
    for i in range(16):
        if sol[i] > 1e-3:
            print(i)
