#!/usr/bin/env python
"""
Shortest path problem
"""

from __future__ import annotations

import contextlib

import numpy as np

with contextlib.suppress(ImportError):
    from coptpy import COPT

from pyepo.model.bases import shortestPathBase
from pyepo.model.copt.coptmodel import _get_envr, optCoptModel
from pyepo.model.utils import _incidence_matrix


class shortestPathModel(shortestPathBase, optCoptModel):
    """
    COPT-backed shortest path on a grid network.

    Attributes:
        _model (COPT model): COPT model
        grid (tuple of int): Size of grid network
        arcs (list): List of arcs
    """

    def _getModel(self) -> tuple:
        """
        A method to build COPT model
        """
        m = _get_envr().createModel("shortest path")
        num_nodes = self.grid[0] * self.grid[1]
        num_arcs = len(self.arcs)
        x = m.addMVar(num_arcs, lb=0.0, ub=1.0, vtype=COPT.CONTINUOUS, nameprefix="x")
        m.setObjSense(COPT.MINIMIZE)
        # sparse node-arc incidence: row v sums (in-flow) - (out-flow) at v
        A = _incidence_matrix(self.arcs, num_nodes)
        b = np.zeros(num_nodes, dtype=np.float64)
        b[0] = -1.0
        b[num_nodes - 1] = 1.0
        m.addConstr(A @ x == b)
        return m, x
