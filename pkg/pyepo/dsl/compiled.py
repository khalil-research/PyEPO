#!/usr/bin/env python
"""
Generic compiled-problem base for the PyEPO DSL.

``compiledBase`` is mixed with a backend base (``optXxxModel``) to form a
concrete compiled problem, e.g. ``compiledGrbProblem(compiledBase,
optGrbModel)``. It carries the backend-agnostic objective handling — scattering
a predicted cost onto its variable positions and solving — while the concrete
subclass builds the solver model and provides the read / write hooks.
"""

from __future__ import annotations

import numpy as np
import torch

from pyepo.model._common import validate_objective_shape
from pyepo.model.opt import optModel
from pyepo.utils import costToNumpy


class compiledBase(optModel):
    """
    Backend-agnostic compiled DSL problem. Mixed with an ``optXxxModel``.
    """

    def __init__(self, problem, params=None):
        # the source DSL Problem and backend solver parameters
        self.problem = problem
        self.params = dict(params) if params else {}
        super().__init__()
        self._apply_params()

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "problem": self.problem,
            "params": self.params,
        }

    @property
    def num_cost(self) -> int:
        # predicted cost dimension
        return self.problem.num_cost

    @property
    def c_pred_index(self):
        # predicted positions
        return self.problem.c_pred_index

    def setObj(self, c):
        """Set the objective from a predicted cost of length ``num_cost``, scattered onto the known fixed costs."""
        prob = self.problem
        coef = costToNumpy(c)
        validate_objective_shape(coef, (prob.num_cost, prob.num_vars))
        # scatter onto fixed costs; an unambiguous full-length vector passes through
        if coef.shape[-1] == prob.num_cost:
            full = prob.fixed_cost.copy()
            full[prob.c_pred_index] += coef
            coef = full
        self._write_obj(coef)

    def _setFullObj(self, c):
        """Set the objective from full-space coefficients (length ``num_vars``), bypassing the predicted-cost scatter."""
        coef = costToNumpy(c)
        validate_objective_shape(coef, self.problem.num_vars, full=True)
        self._write_obj(coef)

    def _fullCost(self, pred_cost):
        # fixed_cost + the predicted cost scattered at its positions (differentiable for torch)
        prob = self.problem
        idx = prob.c_pred_index
        if isinstance(pred_cost, torch.Tensor):
            index = torch.as_tensor(idx, dtype=torch.long, device=pred_cost.device)
            scattered = pred_cost.new_zeros((*pred_cost.shape[:-1], prob.num_vars)).index_add(
                -1, index, pred_cost
            )
            return scattered + torch.as_tensor(
                prob.fixed_cost, dtype=pred_cost.dtype, device=pred_cost.device
            )
        arr = np.asarray(pred_cost, dtype=float)
        full = np.broadcast_to(prob.fixed_cost, (*arr.shape[:-1], prob.num_vars)).copy()
        full[..., idx] += arr
        return full

    def solve(self):
        """Solve and return the full decision-vector solution (length ``num_vars``) with its objective value."""
        sol, obj = self._read_sol()
        # bare objective constants live outside the solver model
        return np.asarray(sol), obj + self.problem.obj_offset

    def addConstr(self, coefs, rhs):
        # add a cut coefs @ x <= rhs over the full variable vector
        coefs = np.asarray(coefs, dtype=float).reshape(-1)
        new_model = self._add_cut(coefs, rhs)
        # track for replay on relax
        new_model._extra_constrs = [*self._extra_constrs, (coefs, float(rhs))]
        return new_model

    def relax(self):
        # recompile the relaxed problem, preserving backend kwargs
        kwargs = self.get_config()
        kwargs["problem"] = self.problem.relax()
        model_rel = type(self)(**kwargs)
        # replay user cuts on the relaxation
        for coefs, rhs in self._extra_constrs:
            model_rel = model_rel.addConstr(coefs, rhs)
        return model_rel

    def _apply_params(self):
        # push self.params to the solver
        pass

    def _write_obj(self, coef):
        # set the full-length objective coefficient vector
        raise NotImplementedError

    def _read_sol(self):
        # optimize and return (full solution, objective value)
        raise NotImplementedError

    def _add_cut(self, coef, rhs):
        # add ``coef @ x <= rhs`` to a fresh copy and return it
        raise NotImplementedError
