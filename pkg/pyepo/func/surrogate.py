#!/usr/bin/env python
"""
Surrogate Loss function
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch.autograd import Function

from pyepo.func._common import is_minimize, validate_positive
from pyepo.func.abcmodule import optModule
from pyepo.func.utils import _solve_or_cache
from pyepo.utils import _EPS

if TYPE_CHECKING:
    from pyepo.data.dataset import optDataset
    from pyepo.func.abcmodule import Reduction
    from pyepo.model.opt import optModel


class SPOPlus(optModule):
    """
    SPO+ loss: a convex surrogate for the SPO regret of a linear-objective LP.

    SPO+ upper-bounds the SPO regret with a convex function of the predicted
    cost vector and provides an informative subgradient (via Danskin's
    theorem) for end-to-end training. It is the strong default for
    predict-then-optimize when true optimal solutions
    :math:`\\mathbf{w}^*(\\mathbf{c})` are available as supervision.

    The forward pass solves the perturbed problem with cost
    :math:`2\\hat{\\mathbf{c}} - \\mathbf{c}` once per training instance and
    returns a scalar loss; the backward pass uses the cached solution to form
    the subgradient :math:`2(\\mathbf{w}^*(\\mathbf{c}) -
    \\mathbf{w}^*(2\\hat{\\mathbf{c}} - \\mathbf{c}))` without any extra
    solver call.

    Reference: Elmachtoub & Grigas (2022)
    `<https://doi.org/10.1287/mnsc.2020.3922>`_
    """

    def __init__(
        self,
        optmodel: optModel,
        processes: int = 1,
        solve_ratio: float = 1.0,
        reduction: Reduction = "mean",
        dataset: optDataset | None = None,
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of instances solved exactly each step (1.0 = no caching)
            reduction: reduction applied to the batch loss (``"mean"``, ``"sum"``, ``"none"``)
            dataset: training dataset used to seed the solution pool when ``solve_ratio < 1``
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)

    def forward(
        self,
        pred_cost: torch.Tensor,
        true_cost: torch.Tensor,
        true_sol: torch.Tensor,
        true_obj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass
        """
        # lift costs to the full objective space (no-op without partial prediction)
        pred_cost = self.optmodel._fullCost(pred_cost)
        true_cost = self.optmodel._fullCost(true_cost)
        loss = cast(
            "torch.Tensor", SPOPlusFunc.apply(pred_cost, true_cost, true_sol, true_obj, self)
        )
        return self._reduce(loss)


class SPOPlusFunc(Function):
    """
    An autograd function for SPO+ Loss
    """

    @staticmethod
    def forward(
        ctx,
        pred_cost: torch.Tensor,
        true_cost: torch.Tensor,
        true_sol: torch.Tensor,
        true_obj: torch.Tensor,
        module: SPOPlus,
    ) -> torch.Tensor:
        """
        Forward pass for SPO+

        Args:
            pred_cost: a batch of predicted values of the cost
            true_cost: a batch of true values of the cost
            true_sol: a batch of true optimal solutions
            true_obj: a batch of true optimal objective values
            module: SPOPlus module

        Returns:
            torch.tensor: SPO+ loss
        """
        # convert tensor
        cp = pred_cost.detach()
        c = true_cost.detach()
        w = true_sol.detach()
        z = true_obj.detach()
        # check sol
        # _check_sol(c, w, z)
        # solve
        sol, obj = _solve_or_cache(2 * cp - c, module)
        # calculate loss
        if is_minimize(module.optmodel.modelSense):
            loss = -obj + 2 * torch.einsum("bi,bi->b", cp, w) - z.squeeze(dim=-1)
        else:
            loss = obj - 2 * torch.einsum("bi,bi->b", cp, w) + z.squeeze(dim=-1)
        # save solutions
        ctx.save_for_backward(true_sol, sol)
        # add other objects to ctx
        ctx.optmodel = module.optmodel
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """
        Backward pass for SPO+
        """
        w, wq = ctx.saved_tensors
        optmodel = ctx.optmodel
        if is_minimize(optmodel.modelSense):
            grad = 2 * (w - wq)
        else:
            grad = 2 * (wq - w)
        return grad_output.unsqueeze(1) * grad, None, None, None, None


class perturbationGradient(optModule):
    """
    Perturbation Gradient (PG): zeroth-order surrogate of the objective-value loss.

    PG approximates the directional derivative of :math:`z^*(\\hat{\\mathbf{c}})`
    along the true cost :math:`\\mathbf{c}` with a finite difference, yielding
    an informative gradient through the otherwise piecewise-constant solver
    layer. Two variants are exposed via ``two_sides``: backward differencing
    (``False``, one extra solve per step) and central differencing (``True``,
    two extra solves but more accurate gradients).

    Unlike SPO+, PG does **not** require true optimal solutions -- it only
    needs the true cost vector :math:`\\mathbf{c}`.

    Reference: Gupta & Huang (2024) `<https://arxiv.org/abs/2402.03256>`_
    """

    def __init__(
        self,
        optmodel: optModel,
        sigma: float = 0.1,
        two_sides: bool = False,
        processes: int = 1,
        solve_ratio: float = 1.0,
        reduction: Reduction = "mean",
        dataset: optDataset | None = None,
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            sigma: finite-difference width (perturbation amplitude)
            two_sides: use central differencing (True) instead of backward (False)
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of instances solved exactly each step (1.0 = no caching)
            reduction: reduction applied to the batch loss (``"mean"``, ``"sum"``, ``"none"``)
            dataset: training dataset used to seed the solution pool when ``solve_ratio < 1``
        """
        validate_positive(sigma, "sigma")
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        # finite difference width
        self.sigma = sigma
        # symmetric perturbation
        self.two_sides = two_sides

    def forward(self, pred_cost: torch.Tensor, true_cost: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # lift costs to the full objective space (no-op without partial prediction)
        pred_cost = self.optmodel._fullCost(pred_cost)
        true_cost = self.optmodel._fullCost(true_cost)
        loss = self._finite_difference(pred_cost, true_cost)
        return self._reduce(loss)

    def _finite_difference(
        self,
        pred_cost: torch.Tensor,
        true_cost: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate the objective-value directional derivative."""
        # convert tensor
        cp = pred_cost.detach()
        c = true_cost.detach()
        # sense-flipped loss sign: MAX problems become minimization
        sign = 1.0 if is_minimize(self.optmodel.modelSense) else -1.0
        b = cp.shape[0]
        # central differencing
        if self.two_sides:
            # batch +sigma and -sigma into one solve
            combined_sol, _ = _solve_or_cache(
                torch.cat([cp + self.sigma * c, cp - self.sigma * c], dim=0),
                self,
            )
            wp, wm = combined_sol[:b], combined_sol[b:]
            # differentiable objective value; the label direction carries no gradient
            obj_plus = torch.einsum("bi,bi->b", pred_cost + self.sigma * c, wp)
            obj_minus = torch.einsum("bi,bi->b", pred_cost - self.sigma * c, wm)
            # loss
            loss = sign * (obj_plus - obj_minus) / (2 * self.sigma + _EPS)
        # back differencing
        else:
            # batch clean and -sigma into one solve
            combined_sol, _ = _solve_or_cache(
                torch.cat([cp, cp - self.sigma * c], dim=0),
                self,
            )
            w, wm = combined_sol[:b], combined_sol[b:]
            # differentiable objective value; the label direction carries no gradient
            obj = torch.einsum("bi,bi->b", pred_cost, w)
            obj_minus = torch.einsum("bi,bi->b", pred_cost - self.sigma * c, wm)
            # loss
            loss = sign * (obj - obj_minus) / (self.sigma + _EPS)
        return loss


# aliases
smartPredictThenOptimizePlus = SPOPlus
PG = perturbationGradient
