#!/usr/bin/env python
"""
Generic compiled-problem base for the PyEPO DSL.

``compiledBase`` is mixed with a backend base (``optXxxModel``) to form a
concrete compiled problem, e.g. ``compiledGrbProblem(compiledBase,
optGrbModel)`` — the same composition as ``knapsackModel(knapsackBase,
optGrbModel)``. It carries the backend-agnostic contract: ``num_cost``,
``setObj`` (broadcast the predicted cost onto its paired solver vars via
``cost_gather``), ``solve`` (reduce each cost group back to cost space),
``addConstr``, and ``relax``. The concrete subclass supplies ``_getModel``,
the cost-paired read/write hooks (``_writeObj`` / ``_writeConstr`` /
``_readCostSol``) over ``_cost_vars``, and ``copy``.
"""

from __future__ import annotations

import numpy as np
import torch

from pyepo.model.opt import optModel
from pyepo.utils import getArgs


class compiledBase(optModel):
    """
    Backend-agnostic compiled DSL problem. Mixed with an ``optXxxModel``.
    """

    @property
    def num_cost(self) -> int:
        # predicted cost dimension = the unique Parameter's size
        return self.problem.cost_param.size

    def setObj(self, c) -> None:
        # broadcast each cost to its paired solver var(s): c[gather] == np.repeat for TSP
        c = c if isinstance(c, (np.ndarray, torch.Tensor)) else np.asarray(c, dtype=np.float32)
        if c.shape[-1] != self.num_cost:
            raise ValueError(f"setObj expected trailing dim {self.num_cost}, got {c.shape[-1]}.")
        g = self.problem.cost_gather
        self._writeObj(c if g is None else c[..., g])

    def solve(self):
        # read the cost-paired vars, then sum each cost group back to cost space
        vals, obj = self._readCostSol()
        g = self.problem.cost_gather
        if g is None:
            return vals, obj
        sol = np.zeros(self.num_cost, dtype=np.float32)
        np.add.at(sol, g, np.asarray(vals, dtype=np.float32))
        return sol, obj

    def addConstr(self, coefs, rhs):
        # broadcast cost-space coefs to the paired vars, then add via the hook
        coefs = np.asarray(coefs, dtype=np.float32).reshape(-1)
        if coefs.size != self.num_cost:
            raise ValueError(f"addConstr expected size {self.num_cost}, got {coefs.size}.")
        g = self.problem.cost_gather
        return self._writeConstr(coefs if g is None else coefs[g], rhs)

    def relax(self):
        # recompile the relaxed problem, preserving backend kwargs
        kwargs = getArgs(self)
        kwargs["problem"] = self.problem.relax()
        return type(self)(**kwargs)
