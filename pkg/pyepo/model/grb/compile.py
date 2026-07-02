#!/usr/bin/env python
"""
Gurobi compiler for the PyEPO DSL.

``compiledGrbProblem`` mixes the shared MVar hooks with ``compiledBase`` and
``optGrbModel`` to turn a finalized DSL ``Problem`` into a GurobiPy model; the
objective handling lives in ``compiledBase``.
"""

from __future__ import annotations

import contextlib

import numpy as np

with contextlib.suppress(ImportError):
    import gurobipy as gp
    from gurobipy import GRB

from pyepo import EPO
from pyepo.dsl.compiled import compiledBase
from pyepo.model._mvar_compile import compiledMVarMixin
from pyepo.model.grb.grbmodel import _require_solution, optGrbModel


def compileProblem(problem, **params) -> compiledGrbProblem:
    """Instantiate the Gurobi-compiled problem; ``params`` are Gurobi parameters."""
    return compiledGrbProblem(problem, params=params)


class compiledGrbProblem(compiledMVarMixin, compiledBase, optGrbModel):
    """
    Gurobi-backed compiled DSL problem.
    """

    def _new_model(self):
        # gurobi model with objective sense applied
        m = gp.Model()
        m.modelSense = GRB.MAXIMIZE if self.problem.modelSense == EPO.MAXIMIZE else GRB.MINIMIZE
        return m

    def _var_specs(self):
        # gurobi infinity, vtype map, and MVar name kwarg
        vtype_map = {
            EPO.BINARY: GRB.BINARY,
            EPO.INTEGER: GRB.INTEGER,
            EPO.CONTINUOUS: GRB.CONTINUOUS,
        }
        return GRB.INFINITY, vtype_map, "name"

    def _read_sol(self):
        # optimize and read the full solution + objective value
        self._model.optimize()
        _require_solution(self._model)
        return np.asarray(self.x.x), self._model.objVal

    def _add_cut(self, coef, rhs):
        # add coef @ x <= rhs to a fresh copy
        new_model = self.copy()
        new_model._model.addConstr(coef @ new_model.x <= float(rhs))
        new_model._model.update()
        return new_model
