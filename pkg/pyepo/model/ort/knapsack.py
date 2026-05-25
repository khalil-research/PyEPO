#!/usr/bin/env python
"""
Knapsack problem
"""

from __future__ import annotations

import numpy as np

try:
    from ortools.linear_solver import pywraplp
    from ortools.sat.python import cp_model
except ImportError:
    pass

from pyepo.model.bases import knapsackBase
from pyepo.model.ort.ortcpmodel import optOrtCpModel
from pyepo.model.ort.ortmodel import optOrtModel


class knapsackModel(knapsackBase, optOrtModel):
    """
    OR-Tools (pywraplp) backed knapsack.

    Attributes:
        _model (pywraplp.Solver): OR-Tools linear solver
        solver (str): solver backend
        weights (np.ndarray): weights of items
        capacity (np.ndarray): total capacity
        items (list): list of item index
    """

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
        x = {i: m.BoolVar(f"x_{i}") for i in self.items}
        # constraints
        for i in range(len(self.capacity)):
            m.Add(
                sum(float(self.weights[i, j]) * x[j] for j in self.items) <= float(self.capacity[i])
            )
        return m, x

    def relax(self) -> knapsackModelRel:
        """
        A method to get linear relaxation model
        """
        return knapsackModelRel(self.weights, self.capacity, self.solver)


class knapsackModelRel(knapsackModel):
    """
    LP relaxation of the OR-Tools knapsack.
    """

    def _getModel(self) -> tuple:
        """
        A method to build OR-Tools pywraplp model (relaxed)

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = pywraplp.Solver.CreateSolver(self.solver.upper())
        if m is None:
            raise RuntimeError(f"Solver '{self.solver}' is not available in OR-Tools.")
        # variables (continuous)
        x = {i: m.NumVar(0, 1, f"x_{i}") for i in self.items}
        # constraints
        for i in range(len(self.capacity)):
            m.Add(
                sum(float(self.weights[i, j]) * x[j] for j in self.items) <= float(self.capacity[i])
            )
        return m, x

    def relax(self) -> knapsackModelRel:
        """
        A forbidden method to relax MIP model
        """
        raise RuntimeError("Model has already been relaxed.")


class knapsackCpModel(knapsackBase, optOrtCpModel):
    """
    OR-Tools CP-SAT backed knapsack.

    Attributes:
        _model (cp_model.CpModel): OR-Tools CP-SAT model
        weights (np.ndarray): weights of items
        capacity (np.ndarray): total capacity
        items (list): list of item index
    """

    def _getModel(self) -> tuple:
        """
        A method to build OR-Tools CP-SAT model

        Returns:
            tuple: optimization model and variables

        Raises:
            ValueError: if weights or capacity contain non-integer values
                (CP-SAT solves over integers; silent truncation would yield
                wrong knapsack solutions).
        """
        weights = self.weights.astype(np.int64)
        capacity = np.asarray(self.capacity).astype(np.int64)
        if not np.array_equal(weights, self.weights):
            raise ValueError(
                "CP-SAT knapsack requires integer weights; got non-integer values. "
                "Cast explicitly, e.g. np.round(weights).astype(int)."
            )
        if not np.array_equal(capacity, np.asarray(self.capacity)):
            raise ValueError("CP-SAT knapsack requires integer capacity; got non-integer values.")
        # create a model
        m = cp_model.CpModel()
        # variables
        x = {i: m.NewBoolVar(f"x_{i}") for i in self.items}
        # constraints
        for i in range(len(capacity)):
            m.Add(sum(int(weights[i, j]) * x[j] for j in self.items) <= int(capacity[i]))
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
    print(f"Obj: {obj}")
    for i in range(16):
        if sol[i] > 1e-3:
            print(i)

    # relax
    print("\n=== pywraplp relaxed ===")
    optmodel_rel = optmodel.relax()
    optmodel_rel.setObj(cost)
    sol, obj = optmodel_rel.solve()
    print(f"Obj: {obj}")
    for i in range(16):
        if sol[i] > 1e-3:
            print(f"{i}: {sol[i]:.4f}")

    # add constraint
    print("\n=== pywraplp + addConstr ===")
    optmodel2 = optmodel.addConstr([1] * 16, 5)
    optmodel2.setObj(cost)
    sol, obj = optmodel2.solve()
    print(f"Obj: {obj}")
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
    print(f"Obj: {obj}")
    for i in range(16):
        if sol[i] > 1e-3:
            print(i)
