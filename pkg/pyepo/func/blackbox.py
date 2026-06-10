#!/usr/bin/env python
"""
Differentiable Black-box optimization function
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch.autograd import Function

from pyepo import EPO
from pyepo.func.abcmodule import optModule
from pyepo.func.utils import _solve_or_cache
from pyepo.utils import _EPS

if TYPE_CHECKING:
    import torch

    from pyepo.data.dataset import optDataset
    from pyepo.model.opt import optModel


class blackboxOpt(optModule):
    """
    Differentiable Black-Box Optimizer (DBB) -- gradient via solution interpolation.

    Replaces the zero gradient of the combinatorial solver with an
    interpolation-based estimate: given an upstream gradient
    :math:`\\mathbf{d}`, DBB approximates the vector-Jacobian product as
    :math:`(\\mathbf{w}^*(\\hat{\\mathbf{c}} + \\lambda \\mathbf{d}) -
    \\mathbf{w}^*(\\hat{\\mathbf{c}})) / \\lambda`. Larger ``lambd`` smooths
    more aggressively; the recommended range is **10-20**. The resulting
    surrogate is nonconvex in :math:`\\hat{\\mathbf{c}}`, so convergence
    guarantees are weaker than SPO+.

    Returns a predicted solution -- pair with an objective-value task loss
    such as L1 against :math:`z^*(\\mathbf{c})`.

    Reference: Vlastelica et al. (2019) `<https://arxiv.org/abs/1912.02175>`_
    """

    def __init__(
        self,
        optmodel: optModel,
        lambd: float = 10,
        processes: int = 1,
        solve_ratio: float = 1.0,
        dataset: optDataset | None = None,
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            lambd: interpolation smoothing strength :math:`\\lambda` (recommended 10-20)
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of instances solved exactly each step (1.0 = no caching)
            dataset: training dataset used to seed the solution pool when ``solve_ratio < 1``
        """
        super().__init__(optmodel, processes, solve_ratio, dataset=dataset)
        # smoothing parameter
        if lambd <= 0:
            raise ValueError("lambda is not positive.")
        self.lambd = lambd

    def forward(self, pred_cost: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # lift costs to the full objective space (no-op without partial prediction)
        pred_cost = self.optmodel._fullCost(pred_cost)
        return cast("torch.Tensor", blackboxOptFunc.apply(pred_cost, self))


class blackboxOptFunc(Function):
    """
    An autograd function for differentiable black-box optimizer
    """

    @staticmethod
    def forward(
        ctx,
        pred_cost: torch.Tensor,
        module: blackboxOpt,
    ) -> torch.Tensor:
        """
        Forward pass for DBB

        Args:
            pred_cost: a batch of predicted values of the cost
            module: blackboxOpt module

        Returns:
            torch.tensor: predicted solutions
        """
        # convert tensor
        cp = pred_cost.detach()
        # solve
        sol, _ = _solve_or_cache(cp, module)
        # save
        ctx.save_for_backward(pred_cost, sol)
        # add other objects to ctx
        ctx.lambd = module.lambd
        ctx.module = module
        return sol

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """
        Backward pass for DBB
        """
        pred_cost, pred_sol = ctx.saved_tensors
        lambd = ctx.lambd
        module = ctx.module
        # convert tensor
        cp = pred_cost.detach()
        wp = pred_sol.detach()
        dl = grad_output.detach()
        # the informative perturbation direction flips for MAX
        if module.optmodel.modelSense == EPO.MINIMIZE:
            sign = 1.0
        else:
            sign = -1.0
        # perturbed costs
        cq = cp + sign * lambd * dl
        # solve
        sol, _ = _solve_or_cache(cq, module)
        # get gradient
        grad = sign * (sol - wp) / (lambd + _EPS)
        return grad, None


class negativeIdentity(optModule):
    """
    Negative Identity Backpropagation (NID) -- hyperparameter-free DBB.

    Treats the solver Jacobian as a (signed) identity:
    :math:`\\partial \\mathbf{w}^* / \\partial \\hat{\\mathbf{c}} \\approx
    -\\mathbf{I}` for minimization (and :math:`+\\mathbf{I}` for
    maximization), yielding a straight-through gradient estimator. This is
    the special case of DBB where :math:`\\lambda` is chosen so the
    interpolated solution coincides with the negative-identity update --
    with the bonus that no extra solver call is needed on the backward
    pass.

    Returns a predicted solution; pair with an objective-value task loss
    (e.g., L1 against :math:`z^*(\\mathbf{c})`).

    Reference: Sahoo et al. (2022) `<https://arxiv.org/abs/2205.15213>`_
    """

    def __init__(
        self,
        optmodel: optModel,
        processes: int = 1,
        solve_ratio: float = 1.0,
        dataset: optDataset | None = None,
    ) -> None:
        """
        Args:
            optmodel: a PyEPO optimization model
            processes: number of solver processes (1 = single-core, 0 = all cores)
            solve_ratio: fraction of instances solved exactly each step (1.0 = no caching)
            dataset: training dataset used to seed the solution pool when ``solve_ratio < 1``
        """
        super().__init__(optmodel, processes, solve_ratio, dataset=dataset)

    def forward(self, pred_cost: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # lift costs to the full objective space (no-op without partial prediction)
        pred_cost = self.optmodel._fullCost(pred_cost)
        return cast("torch.Tensor", negativeIdentityFunc.apply(pred_cost, self))


class negativeIdentityFunc(Function):
    """
    An autograd function for negative identity optimizer
    """

    @staticmethod
    def forward(
        ctx,
        pred_cost: torch.Tensor,
        module: negativeIdentity,
    ) -> torch.Tensor:
        """
        Forward pass for NID

        Args:
            pred_cost: a batch of predicted values of the cost
            module: negativeIdentity module

        Returns:
            torch.tensor: predicted solutions
        """
        # convert tensor
        cp = pred_cost.detach()
        # solve
        sol, _ = _solve_or_cache(cp, module)
        # add other objects to ctx
        ctx.optmodel = module.optmodel
        return sol

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """
        Backward pass for NID
        """
        optmodel = ctx.optmodel
        # negative identity gradient
        if optmodel.modelSense == EPO.MINIMIZE:
            return -grad_output, None
        return grad_output, None


# acronym aliases
DBB = blackboxOpt
NID = negativeIdentity
