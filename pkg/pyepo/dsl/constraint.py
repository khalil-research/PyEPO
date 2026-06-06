#!/usr/bin/env python
"""
Constraint nodes for the PyEPO DSL.

A ``Constraint`` pairs a linear (``Affine``) or quadratic (``Quadratic``) LHS
with a sense and a right-hand side. ``finalize`` absorbs the LHS constant into
the rhs and assembles the global sparse form ``(Q | None, A, sense, b_eff)``.
"""

from __future__ import annotations

import numpy as np


class Constraint:
    """
    A single relation ``lhs (<= | >= | ==) rhs``.

    Attributes:
        lhs (Affine | Quadratic): left-hand side expression
        sense (str): one of ``"<="`` / ``">="`` / ``"=="``
        rhs (np.ndarray): right-hand side, broadcast to the LHS length
    """

    def __init__(self, lhs, sense, rhs):
        self.lhs = lhs
        self.sense = sense
        self.rhs = rhs

    def __bool__(self):
        # guard against accidental truth-testing (e.g. `if expr == rhs`)
        raise TypeError("A Constraint is not a boolean; use it inside Problem(constraints=[...]).")

    def variables(self):
        # Variables referenced by the LHS (encounter order)
        from pyepo.dsl.expression import Quadratic
        if isinstance(self.lhs, Quadratic):
            seen = list(self.lhs.affine.blocks)
            for vi, vj in self.lhs.quad:
                for v in (vi, vj):
                    if v not in seen:
                        seen.append(v)
            return seen
        return list(self.lhs.blocks)

    def has_parameter(self):
        # constraints must be parameter-free; Parameters never reach a Constraint LHS
        from pyepo.dsl.expression import ParametricBilinear
        return isinstance(self.lhs, ParametricBilinear)

    def finalize(self, flat_slice, n_total):
        """
        Assemble the global form ``(Q, A, sense, b_eff)``.

        Returns:
            tuple: ``Q`` (csr ``(n,n)`` or ``None``), ``A`` (csr ``(m,n)``),
                ``sense`` (str), ``b_eff`` (ndarray ``(m,)`` = ``rhs - const``)
        """
        from pyepo.dsl.expression import Affine, Quadratic
        if isinstance(self.lhs, Quadratic):
            aff = self.lhs.affine
            Q = self.lhs.finalize_Q(flat_slice, n_total)
            A = aff.finalize(flat_slice, n_total)
            b_eff = self._rhs_vec(aff.m) - aff.const
            return Q, A, self.sense, b_eff
        aff: Affine = self.lhs
        A = aff.finalize(flat_slice, n_total)
        b_eff = self._rhs_vec(aff.m) - aff.const
        return None, A, self.sense, b_eff

    def _rhs_vec(self, m):
        # broadcast the rhs to a flat (m,) vector
        return np.broadcast_to(np.asarray(self.rhs, dtype=float), (m,)).astype(float)
