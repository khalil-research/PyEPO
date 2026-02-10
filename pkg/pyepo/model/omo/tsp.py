#!/usr/bin/env python
# coding: utf-8
"""
Traveling salesman problem
"""

from collections import defaultdict

import numpy as np
import torch

from pyepo import EPO
from pyepo.model.omo.omomodel import optOmoModel

try:
    from pyomo import opt as po
    from pyomo import environ as pe
    _HAS_PYOMO = True
except ImportError:
    _HAS_PYOMO = False


class tspABModel(optOmoModel):
    """
    This abstract class is an optimization model for the traveling salesman problem.
    This model is for further implementation of different formulation.

    Attributes:
        _model (Pyomo model): Pyomo model
        solver (str): optimization solver in the background
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def __init__(self, num_nodes, solver="glpk"):
        """
        Args:
            num_nodes (int): number of nodes
            solver (str): optimization solver in the background
        """
        self.num_nodes = num_nodes
        self.nodes = list(range(num_nodes))
        self.edges = [(i, j) for i in self.nodes
                      for j in self.nodes if i < j]
        super().__init__(solver)

    @property
    def num_cost(self):
        return len(self.edges)

    def copy(self):
        """
        A method to copy model

        Returns:
            optModel: new copied model
        """
        new_model = type(self)(self.num_nodes, self.solver)
        return new_model

    def getTour(self, sol):
        """
        A method to get a tour from solution

        Args:
            sol (list): solution

        Returns:
            list: a TSP tour
        """
        # active edges
        edges = defaultdict(list)
        for i, (j, k) in enumerate(self.edges):
            if sol[i] > 1e-2:
                edges[j].append(k)
                edges[k].append(j)
        # get tour
        visited = {list(edges.keys())[0]}
        tour = [list(edges.keys())[0]]
        while len(visited) < len(edges):
            i = tour[-1]
            for j in edges[i]:
                if j not in visited:
                    tour.append(j)
                    visited.add(j)
                    break
        if 0 in edges[tour[-1]]:
            tour.append(0)
        return tour


class tspGGModel(tspABModel):
    """
    This class is an optimization model for the traveling salesman problem based on Gavish-Graves (GG) formulation.

    Attributes:
        _model (Pyomo model): Pyomo model
        solver (str): optimization solver in the background
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self):
        """
        A method to build Pyomo model

        Returns:
            tuple: optimization model and variables
        """
        # sense
        self.modelSense = EPO.MINIMIZE
        # create a model
        m = pe.ConcreteModel("tsp")
        # parameters
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        m.dedges = pe.Set(initialize=directed_edges)
        # variables
        x = pe.Var(m.dedges, domain=pe.Binary)
        m.x = x
        y = pe.Var(m.dedges, domain=pe.NonNegativeReals)
        m.y = y
        # constraints
        m.cons = pe.ConstraintList()
        for j in self.nodes:
            m.cons.add(sum(x[i, j] for i in self.nodes if i != j) == 1)
        for i in self.nodes:
            m.cons.add(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for i in self.nodes[1:]:
            m.cons.add(sum(y[i, j] for j in self.nodes if j != i) -
                       sum(y[j, i] for j in self.nodes[1:] if j != i) == 1)
        for (i, j) in directed_edges:
            if i != 0:
                m.cons.add(y[i, j] <= (len(self.nodes) - 1) * x[i, j])
        return m, x

    def setObj(self, c):
        """
        A method to set the objective function

        Args:
            c (list): cost vector
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector does not match number of cost variables.")
        # check if c is a PyTorch tensor
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        else:
            c = np.asarray(c, dtype=np.float32)
        # delete previous component
        self._model.del_component(self._model.obj)
        # set obj
        obj = sum(c[k] * (self.x[i, j] + self.x[j, i])
                  for k, (i, j) in enumerate(self.edges))
        self._model.obj = pe.Objective(sense=pe.minimize, expr=obj)

    def solve(self):
        """
        A method to solve model
        """
        self._solverfac.solve(self._model)
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for k, (i, j) in enumerate(self.edges):
            if pe.value(self.x[i, j]) > 1e-2 or pe.value(self.x[j, i]) > 1e-2:
                sol[k] = 1
        return sol, pe.value(self._model.obj)

    def addConstr(self, coefs, rhs):
        """
        A method to add new constraint

        Args:
            coefs (ndarray): coefficients of new constraint
            rhs (float): right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        if len(coefs) != self.num_cost:
            raise ValueError("Size of coef vector does not match number of cost variables.")
        # copy
        new_model = self.copy()
        # add constraint
        new_model._model.cons.add(
            sum(coefs[k] * (new_model.x[i, j] + new_model.x[j, i])
                for k, (i, j) in enumerate(new_model.edges)) <= rhs)
        return new_model

    def relax(self):
        """
        A method to get linear relaxation model
        """
        # copy
        model_rel = tspGGModelRel(self.num_nodes, self.solver)
        return model_rel


class tspGGModelRel(tspGGModel):
    """
    This class is relaxation of tspGGModel.

    Attributes:
        _model (Pyomo model): Pyomo model
        solver (str): optimization solver in the background
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self):
        """
        A method to build Pyomo model

        Returns:
            tuple: optimization model and variables
        """
        # sense
        self.modelSense = EPO.MINIMIZE
        # create a model
        m = pe.ConcreteModel("tsp")
        # parameters
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        m.dedges = pe.Set(initialize=directed_edges)
        # variables
        x = pe.Var(m.dedges, domain=pe.NonNegativeReals, bounds=(0, 1))
        m.x = x
        y = pe.Var(m.dedges, domain=pe.NonNegativeReals)
        m.y = y
        # constraints
        m.cons = pe.ConstraintList()
        for j in self.nodes:
            m.cons.add(sum(x[i, j] for i in self.nodes if i != j) == 1)
        for i in self.nodes:
            m.cons.add(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for i in self.nodes[1:]:
            m.cons.add(sum(y[i, j] for j in self.nodes if j != i) -
                       sum(y[j, i] for j in self.nodes[1:] if j != i) == 1)
        for (i, j) in directed_edges:
            if i != 0:
                m.cons.add(y[i, j] <= (len(self.nodes) - 1) * x[i, j])
        return m, x

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._solverfac.solve(self._model)
        sol = np.zeros(self.num_cost)
        for k, (i, j) in enumerate(self.edges):
            sol[k] = pe.value(self.x[i, j]) + pe.value(self.x[j, i])
        return sol, pe.value(self._model.obj)

    def relax(self):
        """
        A forbidden method to relax MIP model
        """
        raise RuntimeError("Model has already been relaxed.")

    def getTour(self, sol):
        """
        A forbidden method to get a tour from solution
        """
        raise RuntimeError("Relaxation Model has no integer solution.")


class tspMTZModel(tspABModel):
    """
    This class is an optimization model for the traveling salesman problem based on Miller-Tucker-Zemlin (MTZ) formulation.

    Attributes:
        _model (Pyomo model): Pyomo model
        solver (str): optimization solver in the background
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self):
        """
        A method to build Pyomo model

        Returns:
            tuple: optimization model and variables
        """
        # sense
        self.modelSense = EPO.MINIMIZE
        # create a model
        m = pe.ConcreteModel("tsp")
        # parameters
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        m.dedges = pe.Set(initialize=directed_edges)
        m.nds = pe.Set(initialize=self.nodes)
        # variables
        x = pe.Var(m.dedges, domain=pe.Binary)
        m.x = x
        u = pe.Var(m.nds, domain=pe.NonNegativeReals)
        m.u = u
        # constraints
        m.cons = pe.ConstraintList()
        for j in self.nodes:
            m.cons.add(sum(x[i, j] for i in self.nodes if i != j) == 1)
        for i in self.nodes:
            m.cons.add(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for (i, j) in directed_edges:
            if (i != 0) and (j != 0):
                m.cons.add(u[j] - u[i] >=
                           len(self.nodes) * (x[i, j] - 1) + 1)
        return m, x

    def setObj(self, c):
        """
        A method to set the objective function

        Args:
            c (list): cost vector
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector does not match number of cost variables.")
        # check if c is a PyTorch tensor
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        else:
            c = np.asarray(c, dtype=np.float32)
        # delete previous component
        self._model.del_component(self._model.obj)
        # set obj
        obj = sum(c[k] * (self.x[i, j] + self.x[j, i])
                  for k, (i, j) in enumerate(self.edges))
        self._model.obj = pe.Objective(sense=pe.minimize, expr=obj)

    def solve(self):
        """
        A method to solve model
        """
        self._solverfac.solve(self._model)
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for k, (i, j) in enumerate(self.edges):
            if pe.value(self.x[i, j]) > 1e-2 or pe.value(self.x[j, i]) > 1e-2:
                sol[k] = 1
        return sol, pe.value(self._model.obj)

    def addConstr(self, coefs, rhs):
        """
        A method to add new constraint

        Args:
            coefs (ndarray): coefficients of new constraint
            rhs (float): right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        if len(coefs) != self.num_cost:
            raise ValueError("Size of coef vector does not match number of cost variables.")
        # copy
        new_model = self.copy()
        # add constraint
        new_model._model.cons.add(
            sum(coefs[k] * (new_model.x[i, j] + new_model.x[j, i])
                for k, (i, j) in enumerate(new_model.edges)) <= rhs)
        return new_model

    def relax(self):
        """
        A method to get linear relaxation model
        """
        # copy
        model_rel = tspMTZModelRel(self.num_nodes, self.solver)
        return model_rel


class tspMTZModelRel(tspMTZModel):
    """
    This class is relaxation of tspMTZModel.

    Attributes:
        _model (Pyomo model): Pyomo model
        solver (str): optimization solver in the background
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self):
        """
        A method to build Pyomo model

        Returns:
            tuple: optimization model and variables
        """
        # sense
        self.modelSense = EPO.MINIMIZE
        # create a model
        m = pe.ConcreteModel("tsp")
        # parameters
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        m.dedges = pe.Set(initialize=directed_edges)
        m.nds = pe.Set(initialize=self.nodes)
        # variables
        x = pe.Var(m.dedges, domain=pe.NonNegativeReals, bounds=(0, 1))
        m.x = x
        u = pe.Var(m.nds, domain=pe.NonNegativeReals)
        m.u = u
        # constraints
        m.cons = pe.ConstraintList()
        for j in self.nodes:
            m.cons.add(sum(x[i, j] for i in self.nodes if i != j) == 1)
        for i in self.nodes:
            m.cons.add(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for (i, j) in directed_edges:
            if (i != 0) and (j != 0):
                m.cons.add(u[j] - u[i] >=
                           len(self.nodes) * (x[i, j] - 1) + 1)
        return m, x

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._solverfac.solve(self._model)
        sol = np.zeros(self.num_cost)
        for k, (i, j) in enumerate(self.edges):
            sol[k] = pe.value(self.x[i, j]) + pe.value(self.x[j, i])
        return sol, pe.value(self._model.obj)

    def relax(self):
        """
        A forbidden method to relax MIP model
        """
        raise RuntimeError("Model has already been relaxed.")

    def getTour(self, sol):
        """
        A forbidden method to get a tour from solution
        """
        raise RuntimeError("Relaxation Model has no integer solution.")


if __name__ == "__main__":

    import random
    # random seed
    random.seed(42)
    num_nodes = 5
    num_edges = num_nodes * (num_nodes - 1) // 2
    cost = [random.random() for _ in range(num_edges)]

    # solve GG model
    optmodel = tspGGModel(num_nodes=num_nodes, solver="gurobi")
    optmodel = optmodel.copy()
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print('GG Obj: {}'.format(obj))
    tour = optmodel.getTour(sol)
    print('GG Tour: {}'.format(tour))

    # solve MTZ model
    optmodel = tspMTZModel(num_nodes=num_nodes, solver="gurobi")
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print('MTZ Obj: {}'.format(obj))
    tour = optmodel.getTour(sol)
    print('MTZ Tour: {}'.format(tour))

    # relax GG model
    optmodel = tspGGModel(num_nodes=num_nodes, solver="gurobi")
    optmodel = optmodel.relax()
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print('GG Relaxed Obj: {}'.format(obj))

    # add constraint
    optmodel = tspMTZModel(num_nodes=num_nodes, solver="gurobi")
    optmodel = optmodel.addConstr([1] * num_edges, num_edges - 1)
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print('MTZ + Constr Obj: {}'.format(obj))
