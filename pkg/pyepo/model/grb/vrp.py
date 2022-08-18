#!/usr/bin/env python
# coding: utf-8
"""
Vehicle routing probelm
"""
import copy
from collections import defaultdict

import gurobipy as gp
import numpy as np
import networkx as nx
from gurobipy import GRB

from pyepo.model.grb.grbmodel import optGrbModel


class vrpModel(optGrbModel):
    """
    This class is optimization model for vehicle routing probelm

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
        demands (list(int)): List of customer demands
        capacity (int): Vehicle capacity
        num_vehicle (int): Number of vehicle
    """

    def __init__(self, num_nodes, demands, capacity, num_vehicle):
        """
        Args:
            num_nodes (int): number of nodes
            demands (list(int)): customer demands
            capacity (int): vehicle capacity
            num_vehicle (int): number of vehicle
        """
        self.num_nodes = num_nodes
        self.nodes = list(range(num_nodes))
        self.edges = [(i, j) for i in self.nodes
                      for j in self.nodes if i < j]
        self.demands = self._getDemands(demands)
        self.capacity = capacity
        self.num_vehicle = num_vehicle
        super().__init__()

    def _getDemands(self, d):
        demands_dict = {}
        k = 0
        for v in self.nodes:
            if v == 0:
                continue
            demands_dict[v] = d[k]
            k += 1
        return demands_dict

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("vrp")
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        x = m.addVars([(u,v) for u, v in self.edges if u == 0], name="x", vtype=GRB.INTEGER, ub=2) # single customer
        x.update(m.addVars([(u,v) for u, v in self.edges if u != 0], name="x", vtype=GRB.BINARY))
        for u, v in self.edges:
            x[v, u] = x[u, v]
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstr(x.sum(0, "*") <= 2 * self.num_vehicle) # depot degree
        m.addConstrs(x.sum(v, "*") == 2 for v in self.nodes if v != 0)  # 2 degree
        # activate lazy constraints
        m._x = x
        m._q = self.demands
        m._Q = self.capacity
        m._edges = self.edges
        m.Params.lazyConstraints = 1
        return m, x

    @staticmethod
    def _vrpCallback(model, where):
        """
        A static method to add lazy constraints for VRP
        """
        if where == GRB.Callback.MIPSOL:
            # build corresponding support graph
            edges = []
            for u, v in model._edges:
                if u == 0:
                    continue
                if model.cbGetSolution(model._x[u,v]) > 1e-2:
                    edges.append((u,v))
            gc = nx.Graph()
            gc.add_edges_from(edges)
            # rounded capacity inequalities
            for s in nx.connected_components(gc):
                k = int(np.ceil(np.sum([model._q[v] for v in s]) / model._Q)) # lower bound of required vehicles
                # edges with both end-vertex in S
                edges_s = []
                for u in s:
                    for v in s:
                        if u >= v:
                            continue
                        edges_s.append((u,v))
                if len(s) >= 3:
                    if (len(edges_s) >= len(s)) or (k > 1):
                        model.cbLazy(gp.quicksum(model._x[e] for e in edges_s) <= len(s) - k)

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (list): cost vector
        """
        obj = gp.quicksum(c[i] * self.x[e] for i, e in enumerate(self.edges))
        self._model.setObjective(obj)

    def solve(self):
        """
        A method to solve model
        """
        self._model.update()
        self._model.optimize(self._vrpCallback)
        sol = np.zeros(len(self.edges), dtype=np.uint8)
        for i, e in enumerate(self.edges):
            if self.x[e].x > 1e-2:
                sol[i] = int(np.round(self.x[e].x))
        return sol, self._model.objVal

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
        for i, (u, v) in enumerate(self.edges):
            if sol[i] > 1e-2:
                edges[u].append(v)
                edges[v].append(u)
        # get tour
        route = []
        candidates = edges[0]
        while edges[0]:
            v_curr = 0
            tour = [0]
            v_next = edges[v_curr][0]
            # remove used edges
            edges[v_curr].remove(v_next)
            edges[v_next].remove(v_curr)
            while v_next != 0:
                tour.append(v_next)
                # go to next node
                if not edges[v_next]: # visit single customer
                    v_curr, v_next = v_next, 0
                else:
                    v_curr, v_next = v_next, edges[v_next][0]
                    # remove used edges
                    edges[v_curr].remove(v_next)
                    edges[v_next].remove(v_curr)
            # back to depot
            tour.append(0)
            route.append(tour)
        # check valid
        #for tour in route:
        #    assert (sum(self.demands[v] for v in tour if v != 0)) <= self.capacity, "Infeasible solution"
        return route
