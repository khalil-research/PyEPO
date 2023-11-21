#!/usr/bin/env python
# coding: utf-8
"""
Traveling salesman probelm
"""

from collections import defaultdict
from itertools import combinations

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from pyepo.model.grb.grbmodel import optGrbModel

class unionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i
            return True
        return False


class tspABModel(optGrbModel):
    """
    This abstract class is optimization model for traveling salesman problem.
    This model is for further implementation of different formulation.

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def __init__(self, num_nodes):
        """
        Args:
            num_nodes (int): number of nodes
        """
        self.num_nodes = num_nodes
        self.nodes = list(range(num_nodes))
        self.edges = [(i, j) for i in self.nodes
                      for j in self.nodes if i < j]
        super().__init__()

    @property
    def num_cost(self):
        return len(self.edges)

    def copy(self):
        """
        A method to copy model

        Returns:
            optModel: new copied model
        """
        new_model = type(self)(self.num_nodes)
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
    This class is optimization model for traveling salesman problem based on Gavish–Graves (GG) formulation.

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("tsp")
        # varibles
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, name="x", vtype=GRB.BINARY)
        y = m.addVars(directed_edges, name="y")
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstrs(x.sum("*", j) == 1 for j in self.nodes)
        m.addConstrs(x.sum(i, "*") == 1 for i in self.nodes)
        m.addConstrs(y.sum(i, "*") -
                     gp.quicksum(y[j,i]
                                 for j in self.nodes[1:]
                                 if j != i) == 1
                     for i in self.nodes[1:])
        m.addConstrs(y[i,j] <= (len(self.nodes) - 1) * x[i,j]
                     for (i,j) in x if i != 0)
        return m, x

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (list): cost vector
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector cannot match vars.")
        obj = gp.quicksum(c[k] * (self.x[i,j] + self.x[j,i])
                          for k, (i,j) in enumerate(self.edges))
        self._model.setObjective(obj)

    def solve(self):
        """
        A method to solve model
        """
        self._model.update()
        self._model.optimize()
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for k, (i,j) in enumerate(self.edges):
            if self.x[i,j].x > 1e-2 or self.x[j,i].x > 1e-2:
                sol[k] = 1
        return sol, self._model.objVal

    def addConstr(self, coefs, rhs):
        """
        A method to add new constraint

        Args:
            coefs (ndarray): coeffcients of new constraint
            rhs (float): right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        if len(coefs) != self.num_cost:
            raise ValueError("Size of coef vector cannot cost.")
        # copy
        new_model = self.copy()
        # add constraint
        new_model._model.addConstr(
            gp.quicksum(coefs[k] * (new_model.x[i,j] + new_model.x[j,i])
                        for k, (i,j) in enumerate(new_model.edges)) <= rhs)
        return new_model

    def relax(self):
        """
        A method to get linear relaxation model
        """
        # copy
        model_rel = tspGGModelRel(self.num_nodes)
        return model_rel


class tspGGModelRel(tspGGModel):
    """
    This class is relaxation of tspGGModel.

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("tsp")
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, name="x", ub=1)
        y = m.addVars(directed_edges, name="y")
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstrs(x.sum("*", j) == 1 for j in self.nodes)
        m.addConstrs(x.sum(i, "*") == 1 for i in self.nodes)
        m.addConstrs(y.sum(i, "*") -
                     gp.quicksum(y[j,i]
                                 for j in self.nodes[1:]
                                 if j != i) == 1
                     for i in self.nodes[1:])
        m.addConstrs(y[i,j] <= (len(self.nodes) - 1) * x[i,j]
                     for (i,j) in x if i != 0)
        return m, x

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._model.update()
        self._model.optimize()
        sol = np.zeros(self.num_cost)
        for k, (i,j) in enumerate(self.edges):
            sol[k] = self.x[i,j].x + self.x[j,i].x
        return sol, self._model.objVal

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


class tspDFJModel(tspABModel):
    """
    This class is optimization model for traveling salesman problem based on Danzig–Fulkerson–Johnson (DFJ) formulation and
    constraint generation.

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("tsp")
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        x = m.addVars(self.edges, name="x", vtype=GRB.BINARY)
        for i, j in self.edges:
            x[j, i] = x[i, j]
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstrs(x.sum(i, "*") == 2 for i in self.nodes)  # 2 degree
        # activate lazy constraints
        m._x = x
        m._n = len(self.nodes)
        m.Params.lazyConstraints = 1
        return m, x

    @staticmethod
    def _subtourelim(model, where):
        """
        A static method to add lazy constraints for subtour elimination
        """
        if where == GRB.Callback.MIPSOL:
            # selected edges
            xvals = model.cbGetSolution(model._x)
            selected = gp.tuplelist(
                (i, j) for i, j in model._x.keys() if xvals[i, j] > 1e-2)
            # check subcycle with unionfind
            uf = unionFind(model._n)
            for i, j in selected:
                if not uf.union(i, j):
                    # find subcycle
                    cycle = [k for k in range(model._n) if uf.find(k) == uf.find(i)]
                    if len(cycle) < model._n:
                        constr = gp.quicksum(model._x[i, j]
                                     for i, j in combinations(cycle, 2)) <= len(cycle) - 1
                        model.cbLazy(constr)
                    break

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (list): cost vector
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector cannot match vars.")
        obj = gp.quicksum(c[i] * self.x[k] for i, k in enumerate(self.edges))
        self._model.setObjective(obj)

    def solve(self):
        """
        A method to solve model
        """
        self._model.update()
        self._model.optimize(self._subtourelim)
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for i, e in enumerate(self.edges):
            if self.x[e].x > 1e-2:
                sol[i] = 1
        return sol, self._model.objVal

    def addConstr(self, coefs, rhs):
        """
        A method to add new constraint

        Args:
            coefs (ndarray): coeffcients of new constraint
            rhs (float): right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        if len(coefs) != self.num_cost:
            raise ValueError("Size of coef vector cannot cost.")
        # copy
        new_model = self.copy()
        # add constraint
        new_model._model.addConstr(
            gp.quicksum(coefs[i] * new_model.x[k]
                        for i, k in enumerate(new_model.edges)) <= rhs)
        return new_model


class tspMTZModel(tspABModel):
    """
    This class is optimization model for traveling salesman problem based on Miller-Tucker-Zemlin (MTZ) formulation.

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """
    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("tsp")
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, name="x", vtype=GRB.BINARY)
        u = m.addVars(self.nodes, name="u")
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstrs(x.sum("*", j) == 1 for j in self.nodes)
        m.addConstrs(x.sum(i, "*") == 1 for i in self.nodes)
        m.addConstrs(u[j] - u[i] >=
                     len(self.nodes) * (x[i,j] - 1) + 1
                     for (i,j) in directed_edges
                     if (i != 0) and (j != 0))
        return m, x

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (list): cost vector
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector cannot match vars.")
        obj = gp.quicksum(c[k] * (self.x[i,j] + self.x[j,i])
                          for k, (i,j) in enumerate(self.edges))
        self._model.setObjective(obj)

    def solve(self):
        """
        A method to solve model
        """
        self._model.update()
        self._model.optimize()
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for k, (i,j) in enumerate(self.edges):
            if self.x[i,j].x > 1e-2 or self.x[j,i].x > 1e-2:
                sol[k] = 1
        return sol, self._model.objVal

    def addConstr(self, coefs, rhs):
        """
        A method to add new constraint

        Args:
            coefs (ndarray): coeffcients of new constraint
            rhs (float): right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        if len(coefs) != self.num_cost:
            raise ValueError("Size of coef vector cannot cost.")
        # copy
        new_model = self.copy()
        # add constraint
        new_model._model.addConstr(
            gp.quicksum(coefs[k] * (new_model.x[i,j] + new_model.x[j,i])
                        for k, (i,j) in enumerate(new_model.edges)) <= rhs)
        return new_model

    def relax(self):
        """
        A method to get linear relaxation model
        """
        # copy
        model_rel = tspMTZModelRel(self.num_nodes)
        return model_rel


class tspMTZModelRel(tspMTZModel):
    """
    This class is relaxation of tspMTZModel.

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("tsp")
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, name="x", ub=1)
        u = m.addVars(self.nodes, name="u")
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstrs(x.sum("*", j) == 1 for j in self.nodes)
        m.addConstrs(x.sum(i, "*") == 1 for i in self.nodes)
        m.addConstrs(u[j] - u[i] >=
                     len(self.nodes) * (x[i,j] - 1) + 1
                     for (i,j) in directed_edges
                     if (i != 0) and (j != 0))
        return m, x

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._model.update()
        self._model.optimize()
        sol = np.zeros(self.num_cost)
        for k, (i,j) in enumerate(self.edges):
            sol[k] = self.x[i,j].x + self.x[j,i].x
        return sol, self._model.objVal

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
