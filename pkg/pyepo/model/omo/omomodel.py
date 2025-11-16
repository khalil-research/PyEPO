#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on Pyomo
"""

from copy import copy
import torch
import numpy as np

try:
    from pyomo import opt as po
    from pyomo import environ as pe
    from pyepo import EPO
    from pyepo.model.opt import optModel
    _HAS_PYOMO = True
except ImportError:
    _HAS_PYOMO = False


class optOmoModel(optModel):
    """
    This is an abstract class for Pyomo-based optimization model

    Attributes:
        _model (PyOmo model): Pyomo model
        solver (str): optimization solver in the background
    """

    def __init__(self, solver="glpk"):
        """
        Args:
            solver (str): optimization solver in the background
        """
        super().__init__()
        # error
        if not _HAS_PYOMO:
            raise ImportError("Pyomo is not installed. Please install pyomo to use this feature.")
        # init obj
        if self.modelSense == EPO.MINIMIZE:
            self._model.obj = pe.Objective(sense=pe.minimize, expr=0)
        if self.modelSense == EPO.MAXIMIZE:
            self._model.obj = pe.Objective(sense=pe.maximize, expr=0)
        # set solver
        self.solver = solver
        print("Solver in the background: {}".format(self.solver))
        self._objective_cache = {}
        if self.solver == "gurobi":
            self._solverfac = po.SolverFactory(self.solver, solver_io="python")
        else:
            self._solverfac = po.SolverFactory(self.solver)

    def __repr__(self):
        return "optOmoModel " + self.__class__.__name__
    
    def _objective_fun(self, c):
        """
        Default objective function f(x, c) = cᵀx.
        Can be overridden in subclasses for nonlinear objectives.

        Args:
            c (np.ndarray): cost vector for one sample

        Returns:
            
        """
        return sum(c[i] * self.x[k] for i, k in enumerate(self.x))
        

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (np.ndarray / list): cost of objective function
        """
        # if len(c) != self.num_cost:
        #     raise ValueError("Size of cost vector cannot match vars.")
        # check if c is a PyTorch tensor
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        else:
            c = np.asarray(c, dtype=np.float32)
        # delete previous component
        self._model.del_component(self._model.obj)
        # set obj
        obj = self._objective_fun(c)

        if hasattr(self._model, "obj"):
            self._model.del_component("obj")
        if self.modelSense == EPO.MINIMIZE:
            self._model.obj = pe.Objective(sense=pe.minimize, expr=obj)
        if self.modelSense == EPO.MAXIMIZE:
            self._model.obj = pe.Objective(sense=pe.maximize, expr=obj)

    def _hash_cost(self, c):
        # Convert to a tuple for hashing
        return tuple(np.round(c, decimals=10))  # rounding avoids float precision mismatch

    def setWeightObj(self, w, c):
        """
        Set a weighted objective for predictive prescriptions.

        Args:
            w (np.ndarray): shape (N,), weights for each sample
            c (np.ndarray): shape (N, C), cost vectors for each sample
        """
        # if c.shape[1] != self.num_cost:
        #     raise ValueError("Cost vector dimension mismatch.")
        if c.shape[0] != len(w):
            raise ValueError("Weights and costs must have same first dimension.")
        
        # Check if PyTorch tensor inputs
        if isinstance(w, torch.Tensor):
            w = w.detach().cpu().numpy()
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        
        # Build or retrieve objective terms from cache
        obj_terms = []
        for i in range(len(w)):
            key = self._hash_cost(c[i])
            if key not in self._objective_cache:
                self._objective_cache[key] = self._objective_fun(c[i])
                # print("Caching objective for cost:", c[i])
            obj_terms.append(w[i] * self._objective_cache[key])

        obj = pe.quicksum(obj_terms)

        # obj = pe.quicksum(w[i] * self._objective_fun(c[i]) for i in range(len(w)))
        
        if hasattr(self._model, "obj"):
            self._model.del_component("obj")
        if self.modelSense == EPO.MINIMIZE:
            self._model.obj = pe.Objective(sense=pe.minimize, expr=obj)
        if self.modelSense == EPO.MAXIMIZE:
            self._model.obj = pe.Objective(sense=pe.maximize, expr=obj)

    def cal_obj(self, c, x):
        """"
        An abstract method to calculate the objective value

        Args:
            c (ndarray): cost of objective
            x (ndarray): the decision variables
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector cannot match vars.")

        # check if c is a PyTorch tensor
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        else:
            c = np.asarray(c, dtype=np.float32)

        # check if c is a PyTorch tensor
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x, dtype=np.float32)

        obj = c @ x

        return obj

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        # solve
        self._solverfac.solve(self._model)
        return [pe.value(self.x[k]) for k in self.x], pe.value(self._model.obj)

    def copy(self):
        """
        A method to copy model

        Returns:
            optModel: new copied model
        """
        new_model = copy(self)
        # new model
        new_model._model = self._model.clone()
        # variables for new model
        new_model.x = new_model._model.x
        return new_model

    def addConstr(self, coefs, rhs):
        """
        A method to add new constraint

        Args:
            coefs (np.ndarray / list): coeffcients of new constraint
            rhs (float): right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        if len(coefs) != self.num_cost:
            raise ValueError("Size of coef vector cannot cost.")
        # copy
        new_model = self.copy()
        # add constraint
        expr = sum(coefs[i] * new_model.x[k]
                   for i, k in enumerate(new_model.x)) <= rhs
        new_model._model.cons.add(expr)
        return new_model
