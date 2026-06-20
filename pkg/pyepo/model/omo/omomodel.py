#!/usr/bin/env python
"""
Abstract optimization model based on Pyomo
"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import numpy as np

from pyepo import EPO
from pyepo.model._common import validate_objective_shape
from pyepo.model.opt import optModel
from pyepo.utils import costToNumpy

try:
    from pyomo import environ as pe
    from pyomo import opt as po

    _HAS_PYOMO = True
except ImportError:
    _HAS_PYOMO = False

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


class optOmoModel(optModel):
    """
    Abstract base class for Pyomo-backed optimization models.

    Subclasses implement ``_getModel`` to build a Pyomo ``ConcreteModel``
    and return ``(model, variables)``. Unlike ``optGrbModel``, the objective
    sense is **not** detected automatically -- set ``self.modelSense =
    EPO.MAXIMIZE`` in ``_getModel`` for maximization problems (default is
    minimization). The cost vector is wired into the model as a mutable
    ``Param`` so that ``setObj`` only updates parameter values rather than
    rebuilding the objective expression.

    Any solver supported by Pyomo can be plugged in via the ``solver``
    argument (e.g., ``"glpk"``, ``"gurobi"``, ``"cbc"``).

    Attributes:
        _model (pyomo.ConcreteModel): underlying Pyomo model
        solver (str): name of the Pyomo solver backend
    """

    def __init__(self, solver: str = "glpk") -> None:
        """
        Args:
            solver: name of the Pyomo solver backend (e.g. ``"glpk"``, ``"gurobi"``)
        """
        # error
        if not _HAS_PYOMO:
            raise ImportError("Pyomo is not installed. Please install pyomo to use this feature.")
        super().__init__()
        self._model.cost = pe.Param(range(self.num_cost), mutable=True, initialize=0.0)
        if self.modelSense == EPO.MINIMIZE:
            sense = pe.minimize
        elif self.modelSense == EPO.MAXIMIZE:
            sense = pe.maximize
        else:
            raise ValueError("Invalid modelSense.")
        self._model.obj = pe.Objective(expr=self._obj_expr(), sense=sense)
        # set solver
        self.solver = solver
        if self.solver == "gurobi":
            self._solverfac = po.SolverFactory(self.solver, solver_io="python")
        else:
            self._solverfac = po.SolverFactory(self.solver)

    def __repr__(self) -> str:
        return "optOmoModel " + self.__class__.__name__

    def get_config(self) -> dict:
        return {**super().get_config(), "solver": self.solver}

    def _obj_expr(self):
        """Parameterized objective expression. Override for non-trivial variable groupings (e.g., TSP paired edges)."""
        return sum(self._model.cost[i] * self.x[k] for i, k in enumerate(self.x))

    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        """
        A method to set the objective function

        Args:
            c: cost of objective function
        """
        validate_objective_shape(c, self.num_cost)
        c = costToNumpy(c)
        for i in range(self.num_cost):
            self._model.cost[i] = float(c[i])

    def solve(self) -> tuple[np.ndarray, float]:
        """
        A method to solve the model

        Returns:
            tuple: optimal solution (np.ndarray) and objective value (float)
        """
        res = self._solverfac.solve(self._model)
        # surface failed solves clearly instead of an uninitialized-value error
        cond = res.solver.termination_condition
        if cond in (
            po.TerminationCondition.infeasible,
            po.TerminationCondition.unbounded,
            po.TerminationCondition.infeasibleOrUnbounded,
            po.TerminationCondition.error,
        ):
            raise RuntimeError(f"Pyomo found no solution (termination {cond}).")
        sol = np.fromiter(
            (pe.value(self.x[k]) for k in self.x),
            dtype=np.float32,
        )
        return sol, float(pe.value(self._model.obj))

    def copy(self) -> Self:
        """
        A method to copy the model

        Returns:
            optModel: new copied model
        """
        new_model = copy(self)
        # new model
        new_model._model = self._model.clone()
        # variables for new model
        new_model.x = new_model._model.x
        return new_model

    def addConstr(self, coefs: np.ndarray | torch.Tensor | list, rhs: float) -> Self:
        """
        A method to add a new constraint

        Args:
            coefs: coefficients of new constraint
            rhs: right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        if len(coefs) != self.num_cost:
            raise ValueError("Size of coef vector does not match number of cost variables.")
        # copy
        new_model = self.copy()
        # add constraint
        expr = sum(coefs[i] * new_model.x[k] for i, k in enumerate(new_model.x)) <= rhs
        new_model._model.cons.add(expr)
        # track for replay on relax
        new_model._extra_constrs = [*self._extra_constrs, (costToNumpy(coefs), float(rhs))]
        return new_model
