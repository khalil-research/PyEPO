#!/usr/bin/env python
"""Tests for the DSL Gurobi backend (pyepo.dsl -> compiledGrbProblem).

The Phase 2 Gurobi compiler turns a finalized DSL Problem into a GurobiPy
model. The oracle is golden equivalence to the hand-written legacy classes:
the same predicted cost must give the same objective and solution. Quadratic
constraints, LP relaxation, and the optModel contract (for pyepo.func) are
also checked. Skips when Gurobi is unavailable.
"""

import numpy as np
import pytest

from pyepo import dsl
from pyepo.model.opt import optModel

from .conftest import requires_gurobi


@requires_gurobi
def test_knapsack_matches_legacy():
    import pyepo.model.grb as grb
    W = np.array([[3.0, 4, 3, 6, 4], [4, 5, 2, 3, 5], [5, 4, 6, 2, 3]])
    cap = np.array([12.0, 10, 15])
    legacy = grb.knapsackModel(W, cap)

    x = dsl.Variable(5, binary=True)
    c = dsl.Parameter(5)
    comp = dsl.Problem(dsl.Maximize(c @ x), [W @ x <= cap]).compile(backend="gurobi")

    assert isinstance(comp, optModel) and comp.num_cost == 5
    rng = np.random.default_rng(0)
    for _ in range(50):
        cost = rng.standard_normal(5)
        legacy.setObj(cost)
        sol_l, obj_l = legacy.solve()
        comp.setObj(cost)
        sol_d, obj_d = comp.solve()
        assert obj_d == pytest.approx(obj_l, abs=1e-6)
        assert np.allclose(np.asarray(sol_d), np.asarray(sol_l), atol=1e-6)


@requires_gurobi
def test_relax_matches_legacy_lp():
    import pyepo.model.grb as grb
    W = np.array([[3.0, 4, 3, 6, 4], [4, 5, 2, 3, 5]])
    cap = np.array([9.0, 8.0])
    legacy_lp = grb.knapsackModel(W, cap).relax()

    x = dsl.Variable(5, binary=True)
    c = dsl.Parameter(5)
    comp_lp = dsl.Problem(dsl.Maximize(c @ x), [W @ x <= cap]).compile(backend="gurobi").relax()

    assert not comp_lp.problem.var_binary.any()             # relaxed to continuous
    rng = np.random.default_rng(1)
    for _ in range(20):
        cost = rng.standard_normal(5)
        legacy_lp.setObj(cost)
        _, obj_l = legacy_lp.solve()
        comp_lp.setObj(cost)
        _, obj_d = comp_lp.solve()
        assert obj_d == pytest.approx(obj_l, abs=1e-6)


@requires_gurobi
def test_portfolio_quadratic_constraint_solves():
    rng = np.random.default_rng(2)
    cov = np.cov(rng.standard_normal((60, 5)), rowvar=False)
    budget = 2.25 * cov.mean()
    x = dsl.Variable(5, lb=0)
    c = dsl.Parameter(5)
    comp = dsl.Problem(dsl.Maximize(c @ x),
                       [x.sum() == 1, dsl.quadForm(x, cov) <= budget]).compile(backend="gurobi")
    comp.setObj(rng.standard_normal(5))
    sol, _ = comp.solve()
    assert sol.sum() == pytest.approx(1.0, abs=1e-5)        # budget constraint
    assert sol @ ((cov + cov.T) / 2) @ sol <= budget + 1e-6  # risk constraint
    assert (sol >= -1e-6).all()                             # lb = 0


@requires_gurobi
def test_assignment_solves_to_permutation():
    n = 4
    x = dsl.Variable((n, n), binary=True)
    c = dsl.Parameter((n, n))
    comp = dsl.Problem(dsl.Minimize(dsl.sum(c * x)),
                       [x.sum(axis=1) == 1, x.sum(axis=0) == 1]).compile(backend="gurobi")
    rng = np.random.default_rng(3)
    comp.setObj(rng.standard_normal(n * n))
    sol, _ = comp.solve()
    P = np.asarray(sol).reshape(n, n)
    assert np.allclose(P.sum(axis=0), 1) and np.allclose(P.sum(axis=1), 1)   # permutation


@requires_gurobi
def test_addconstr_cost_space():
    W = np.array([[3.0, 4, 3, 6, 4]])
    cap = np.array([9.0])
    x = dsl.Variable(5, binary=True)
    c = dsl.Parameter(5)
    comp = dsl.Problem(dsl.Maximize(c @ x), [W @ x <= cap]).compile(backend="gurobi")
    # add a cost-space cut: sum of first three items <= 1
    comp2 = comp.addConstr([1, 1, 1, 0, 0], 1.0)
    comp2.setObj(np.ones(5))
    sol, _ = comp2.solve()
    assert np.asarray(sol)[:3].sum() <= 1 + 1e-6
