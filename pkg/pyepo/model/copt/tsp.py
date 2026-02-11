#!/usr/bin/env python
# coding: utf-8
"""
Traveling salesman problem
"""

from collections import defaultdict
from itertools import combinations

import numpy as np
from coptpy import Envr
from coptpy import COPT
from coptpy import CallbackBase

from pyepo.model.copt.coptmodel import optCoptModel
from pyepo.model.opt import unionFind


class tspABModel(optCoptModel):
    """
    This abstract class is an optimization model for the traveling salesman problem.
    This model is for further implementation of different formulation.

    Attributes:
        _model (COPT model): COPT model
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
            else:
                # all neighbors visited: disconnected graph, stop
                break
        if 0 in edges[tour[-1]]:
            tour.append(0)
        return tour


class tspGGModel(tspABModel):
    """
    This class is an optimization model for the traveling salesman problem based on Gavish-Graves (GG) formulation.

    Attributes:
        _model (COPT model): COPT model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self):
        """
        A method to build COPT model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = Envr().createModel("tsp")
        # variables
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, nameprefix='x', vtype=COPT.BINARY)
        y = m.addVars(directed_edges, nameprefix='y', vtype=COPT.CONTINUOUS)
        # sense
        m.setObjSense(COPT.MINIMIZE)
        # constraints
        for j in self.nodes:
            m.addConstr(sum(x[i, j] for i in self.nodes if i != j) == 1)
        for i in self.nodes:
            m.addConstr(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for i in self.nodes[1:]:
            m.addConstr(sum(y[i, j] for j in self.nodes if j != i) -
                        sum(y[j, i] for j in self.nodes[1:] if j != i) == 1)
        for (i, j) in directed_edges:
            if i != 0:
                m.addConstr(y[i, j] <= (len(self.nodes) - 1) * x[i, j])
        return m, x

    def setObj(self, c):
        """
        A method to set the objective function

        Args:
            c (list): cost vector
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector does not match number of cost variables.")
        obj = sum(c[k] * (self.x[i, j] + self.x[j, i])
                  for k, (i, j) in enumerate(self.edges))
        self._model.setObjective(obj)

    def solve(self):
        """
        A method to solve model
        """
        self._model.solve()
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for k, (i, j) in enumerate(self.edges):
            if self.x[i, j].x > 1e-2 or self.x[j, i].x > 1e-2:
                sol[k] = 1
        return sol, self._model.objVal

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
        new_model._model.addConstr(
            sum(coefs[k] * (new_model.x[i, j] + new_model.x[j, i])
                for k, (i, j) in enumerate(new_model.edges)) <= rhs)
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
        _model (COPT model): COPT model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self):
        """
        A method to build COPT model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = Envr().createModel("tsp")
        # variables
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, nameprefix='x', vtype=COPT.CONTINUOUS,
                       lb=0, ub=1)
        y = m.addVars(directed_edges, nameprefix='y', vtype=COPT.CONTINUOUS)
        # sense
        m.setObjSense(COPT.MINIMIZE)
        # constraints
        for j in self.nodes:
            m.addConstr(sum(x[i, j] for i in self.nodes if i != j) == 1)
        for i in self.nodes:
            m.addConstr(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for i in self.nodes[1:]:
            m.addConstr(sum(y[i, j] for j in self.nodes if j != i) -
                        sum(y[j, i] for j in self.nodes[1:] if j != i) == 1)
        for (i, j) in directed_edges:
            if i != 0:
                m.addConstr(y[i, j] <= (len(self.nodes) - 1) * x[i, j])
        return m, x

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._model.solve()
        sol = np.zeros(self.num_cost)
        for k, (i, j) in enumerate(self.edges):
            sol[k] = self.x[i, j].x + self.x[j, i].x
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
    This class is an optimization model for the traveling salesman problem based on Danzig-Fulkerson-Johnson (DFJ) formulation and
    constraint generation.

    Attributes:
        _model (COPT model): COPT model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    class _SubtourCallback(CallbackBase):
        """
        A callback class for subtour elimination
        """
        def __init__(self, x, n, edges):
            super().__init__()
            self._x = x
            self._n = n
            self._edges = edges

        def callback(self):
            if self.where() == COPT.CBCONTEXT_MIPSOL:
                # selected edges
                xvals = self.getSolution(self._x)
                selected = [(i, j) for i, j in self._x.keys()
                            if xvals[i, j] > 1e-2]
                # check subcycle with unionfind
                uf = unionFind(self._n)
                for i, j in selected:
                    if not uf.union(i, j):
                        # find subcycle
                        cycle = [k for k in range(self._n)
                                 if uf.find(k) == uf.find(i)]
                        if len(cycle) < self._n:
                            constr = sum(self._x[i, j]
                                         for i, j in combinations(cycle, 2))
                            self.addLazyConstr(constr <= len(cycle) - 1)
                        break

    def _getModel(self):
        """
        A method to build COPT model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = Envr().createModel("tsp")
        # variables
        x = m.addVars(self.edges, nameprefix='x', vtype=COPT.BINARY)
        for i, j in self.edges:
            x[j, i] = x[i, j]
        # sense
        m.setObjSense(COPT.MINIMIZE)
        # constraints
        for i in self.nodes:
            m.addConstr(sum(x[i, j] for j in self.nodes if j != i) == 2)
        return m, x

    def setObj(self, c):
        """
        A method to set the objective function

        Args:
            c (list): cost vector
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector does not match number of cost variables.")
        obj = sum(c[i] * self.x[k] for i, k in enumerate(self.edges))
        self._model.setObjective(obj)

    def solve(self):
        """
        A method to solve model
        """
        cb = self._SubtourCallback(self.x, len(self.nodes), self.edges)
        self._model.setCallback(cb, COPT.CBCONTEXT_MIPSOL)
        self._model.solve()
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for i, e in enumerate(self.edges):
            if self.x[e].x > 1e-2:
                sol[i] = 1
        return sol, self._model.objVal

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
        new_model._model.addConstr(
            sum(coefs[i] * new_model.x[k]
                for i, k in enumerate(new_model.edges)) <= rhs)
        return new_model


class tspMTZModel(tspABModel):
    """
    This class is an optimization model for the traveling salesman problem based on Miller-Tucker-Zemlin (MTZ) formulation.

    Attributes:
        _model (COPT model): COPT model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self):
        """
        A method to build COPT model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = Envr().createModel("tsp")
        # variables
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, nameprefix='x', vtype=COPT.BINARY)
        u = m.addVars(self.nodes, nameprefix='u', vtype=COPT.CONTINUOUS)
        # sense
        m.setObjSense(COPT.MINIMIZE)
        # constraints
        for j in self.nodes:
            m.addConstr(sum(x[i, j] for i in self.nodes if i != j) == 1)
        for i in self.nodes:
            m.addConstr(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for (i, j) in directed_edges:
            if (i != 0) and (j != 0):
                m.addConstr(u[j] - u[i] >=
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
        obj = sum(c[k] * (self.x[i, j] + self.x[j, i])
                  for k, (i, j) in enumerate(self.edges))
        self._model.setObjective(obj)

    def solve(self):
        """
        A method to solve model
        """
        self._model.solve()
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for k, (i, j) in enumerate(self.edges):
            if self.x[i, j].x > 1e-2 or self.x[j, i].x > 1e-2:
                sol[k] = 1
        return sol, self._model.objVal

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
        new_model._model.addConstr(
            sum(coefs[k] * (new_model.x[i, j] + new_model.x[j, i])
                for k, (i, j) in enumerate(new_model.edges)) <= rhs)
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
        _model (COPT model): COPT model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """

    def _getModel(self):
        """
        A method to build COPT model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = Envr().createModel("tsp")
        # variables
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, nameprefix='x', vtype=COPT.CONTINUOUS,
                       lb=0, ub=1)
        u = m.addVars(self.nodes, nameprefix='u', vtype=COPT.CONTINUOUS)
        # sense
        m.setObjSense(COPT.MINIMIZE)
        # constraints
        for j in self.nodes:
            m.addConstr(sum(x[i, j] for i in self.nodes if i != j) == 1)
        for i in self.nodes:
            m.addConstr(sum(x[i, j] for j in self.nodes if j != i) == 1)
        for (i, j) in directed_edges:
            if (i != 0) and (j != 0):
                m.addConstr(u[j] - u[i] >=
                            len(self.nodes) * (x[i, j] - 1) + 1)
        return m, x

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._model.solve()
        sol = np.zeros(self.num_cost)
        for k, (i, j) in enumerate(self.edges):
            sol[k] = self.x[i, j].x + self.x[j, i].x
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


if __name__ == "__main__":

    import random
    # random seed
    random.seed(42)
    num_nodes = 5
    num_edges = num_nodes * (num_nodes - 1) // 2
    cost = [random.random() for _ in range(num_edges)]

    # solve GG model
    optmodel = tspGGModel(num_nodes=num_nodes)
    optmodel = optmodel.copy()
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print('GG Obj: {}'.format(obj))
    tour = optmodel.getTour(sol)
    print('GG Tour: {}'.format(tour))

    # solve DFJ model
    optmodel = tspDFJModel(num_nodes=num_nodes)
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print('DFJ Obj: {}'.format(obj))
    tour = optmodel.getTour(sol)
    print('DFJ Tour: {}'.format(tour))

    # solve MTZ model
    optmodel = tspMTZModel(num_nodes=num_nodes)
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print('MTZ Obj: {}'.format(obj))
    tour = optmodel.getTour(sol)
    print('MTZ Tour: {}'.format(tour))

    # relax GG model
    optmodel = tspGGModel(num_nodes=num_nodes)
    optmodel = optmodel.relax()
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print('GG Relaxed Obj: {}'.format(obj))

    # add constraint
    optmodel = tspMTZModel(num_nodes=num_nodes)
    optmodel = optmodel.addConstr([1] * num_edges, num_edges - 1)
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print('MTZ + Constr Obj: {}'.format(obj))
