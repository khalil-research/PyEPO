#!/usr/bin/env python
"""
COPT compiler for the PyEPO DSL.

``compiledCoptProblem`` mixes the shared MVar hooks with ``compiledBase`` and
``optCoptModel`` to turn a finalized DSL ``Problem`` into a Cardinal Optimizer
model; the objective handling lives in ``compiledBase``.
"""

from __future__ import annotations

import contextlib

import numpy as np

with contextlib.suppress(ImportError):
    from coptpy import COPT

from pyepo import EPO
from pyepo.dsl.compiled import compiledBase
from pyepo.model._mvar_compile import compiledMVarMixin
from pyepo.model.copt.coptmodel import _get_envr, _read_solution, optCoptModel


def compileProblem(problem, **params) -> compiledCoptProblem:
    """Instantiate the COPT-compiled problem; ``params`` are COPT parameters."""
    return compiledCoptProblem(problem, params=params)


class compiledCoptProblem(compiledMVarMixin, compiledBase, optCoptModel):
    """
    COPT-backed compiled DSL problem.
    """

    def _new_model(self):
        # COPT model with objective sense applied
        m = _get_envr().createModel("dsl")
        m.setObjSense(COPT.MAXIMIZE if self.problem.modelSense == EPO.MAXIMIZE else COPT.MINIMIZE)
        return m

    def _var_specs(self):
        # COPT infinity, vtype map, and MVar name kwarg
        vtype_map = {
            EPO.BINARY: COPT.BINARY,
            EPO.INTEGER: COPT.INTEGER,
            EPO.CONTINUOUS: COPT.CONTINUOUS,
        }
        return COPT.INFINITY, vtype_map, "nameprefix"

    def _quad_adder(self, m):
        # quadratic constraints go through addQConstr
        return m.addQConstr

    def _read_sol(self):
        # optimize and read the full solution + objective value
        self._model.solve()
        return _read_solution(
            self._model,
            lambda: np.asarray(self.x.x.tolist(), dtype=float),
        )

    def _add_cut(self, coef, rhs):
        # add coef @ x <= rhs to a fresh copy
        new_model = self.copy()
        new_model._model.addConstr(coef @ new_model.x <= float(rhs))
        return new_model
