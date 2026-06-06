#!/usr/bin/env python
"""Tests for pyepo.dsl: the symbolic DSL core (Phase 1).

The DSL core is pure numpy/scipy and needs no solver backend. These tests
check that the finalized IR (global sparse coefficient matrices, cost layout)
matches hand-derived forms for the canonical problem shapes (knapsack,
shortest path, portfolio, assignment, TSP-MTZ broadcast), that the expression
operators are correct, and that the PyEPO-scope validation fires.
"""

import numpy as np
import pytest

from pyepo import dsl

# ============================================================
# Variable / Parameter construction
# ============================================================

def test_variable_binary_bounds():
    x = dsl.Variable(4, binary=True)
    assert x.size == 4 and x.binary
    assert np.allclose(x.lb, 0) and np.allclose(x.ub, 1)


def test_variable_integer_binary_exclusive():
    with pytest.raises(ValueError):
        dsl.Variable(3, integer=True, binary=True)


def test_variable_multidim_and_bounds_broadcast():
    x = dsl.Variable((2, 3), lb=-1.0, ub=2.0)
    assert x.shape == (2, 3) and x.size == 6
    assert np.allclose(x.lb, -1.0) and np.allclose(x.ub, 2.0)


# ============================================================
# Affine algebra vs hand-derived sparse coefficients
# ============================================================

def test_matmul_and_const_constraint():
    W = np.array([[3.0, 4, 3, 6, 4], [4, 5, 2, 3, 5]])
    cap = np.array([12.0, 10.0])
    x = dsl.Variable(5)
    con = (W @ x <= cap)
    Q, A, sense, b = con.finalize({x: slice(0, 5)}, 5)
    assert Q is None and sense == "<="
    assert np.allclose(A.toarray(), W)
    assert np.allclose(b, cap)


def test_affine_offset_absorbed_into_rhs():
    x = dsl.Variable(3)
    con = (x.sum() + 2.0 == 5.0)            # x0+x1+x2 + 2 == 5  ->  ... == 3
    _, A, sense, b = con.finalize({x: slice(0, 3)}, 3)
    assert sense == "==" and np.allclose(A.toarray(), np.ones((1, 3)))
    assert np.allclose(b, 3.0)


def test_sum_axes_match_kron():
    x = dsl.Variable((3, 3))
    _, Arow, _, _ = (x.sum(axis=1) == 1).finalize({x: slice(0, 9)}, 9)
    _, Acol, _, _ = (x.sum(axis=0) == 1).finalize({x: slice(0, 9)}, 9)
    assert np.allclose(Arow.toarray(), np.kron(np.eye(3), np.ones((1, 3))))
    assert np.allclose(Acol.toarray(), np.kron(np.ones((1, 3)), np.eye(3)))


def test_indexing_selects_rows():
    x = dsl.Variable(5)
    _, A, _, _ = (x[[0, 2, 4]] <= 1).finalize({x: slice(0, 5)}, 5)
    expected = np.zeros((3, 5))
    expected[0, 0] = expected[1, 2] = expected[2, 4] = 1.0
    assert np.allclose(A.toarray(), expected)


def test_scalar_scale_and_subtract():
    x = dsl.Variable(3)
    _, A, _, b = (2.0 * x - 1.0 <= 4.0).finalize({x: slice(0, 3)}, 3)
    assert np.allclose(A.toarray(), 2.0 * np.eye(3))
    assert np.allclose(b, 5.0)             # rhs 4 - const(-1) = 5


# ============================================================
# Full problems: finalized IR vs hand-derived
# ============================================================

def test_knapsack_ir():
    W = np.array([[3.0, 4, 3, 6, 4], [4, 5, 2, 3, 5], [5, 4, 6, 2, 3]])
    cap = np.array([12.0, 10, 15])
    x = dsl.Variable(5, binary=True)
    c = dsl.Parameter(5)
    prob = dsl.Problem(dsl.Maximize(c @ x), [W @ x <= cap])
    assert prob.n_total == 5 and prob.num_cost == 5
    assert prob.objective.sense == "maximize"
    Q, A, sense, b = prob.constraints_ir[0]
    assert Q is None and sense == "<="
    assert np.allclose(A.toarray(), W) and np.allclose(b, cap)
    assert prob.cost_gather is None                # 1:1 cost, no gather
    assert prob.var_binary.all()


def test_shortestpath_equality_ir():
    # generic flow-conservation equality A_eq x = b
    A_eq = np.array([[1.0, 1, 0, 0], [-1, 0, 1, 0], [0, -1, -1, 1]])
    b_eq = np.array([1.0, 0, -1])
    x = dsl.Variable(4)
    c = dsl.Parameter(4)
    prob = dsl.Problem(dsl.Minimize(c @ x), [A_eq @ x == b_eq])
    _, A, sense, b = prob.constraints_ir[0]
    assert sense == "==" and np.allclose(A.toarray(), A_eq) and np.allclose(b, b_eq)
    assert prob.objective.epo_sense.name == "MINIMIZE"


def test_assignment_ir():
    x = dsl.Variable((3, 3), binary=True)
    c = dsl.Parameter((3, 3))
    prob = dsl.Problem(dsl.Minimize(dsl.sum(c * x)),
                       [x.sum(axis=1) == 1, x.sum(axis=0) == 1])
    assert prob.num_cost == 9
    _, Ar, _, _ = prob.constraints_ir[0]
    _, Ac, _, _ = prob.constraints_ir[1]
    assert np.allclose(Ar.toarray(), np.kron(np.eye(3), np.ones((1, 3))))
    assert np.allclose(Ac.toarray(), np.kron(np.ones((1, 3)), np.eye(3)))
    assert prob.cost_gather is None               # 1:1 cost, no gather


def test_portfolio_quadratic_constraint_ir():
    rng = np.random.default_rng(0)
    cov = np.cov(rng.standard_normal((50, 5)), rowvar=False)
    budget = 2.25 * cov.mean()
    x = dsl.Variable(5, lb=0)
    c = dsl.Parameter(5)
    prob = dsl.Problem(dsl.Maximize(c @ x),
                       [x.sum() == 1, dsl.quadForm(x, cov) <= budget])
    _, As, _, bs = prob.constraints_ir[0]
    Qq, Aq, sq, bq = prob.constraints_ir[1]
    assert np.allclose(As.toarray(), np.ones((1, 5))) and np.allclose(bs, 1.0)
    assert sq == "<=" and Qq is not None
    assert np.allclose(Qq.toarray(), (cov + cov.T) / 2)     # symmetric quadratic part
    assert np.allclose(Aq.toarray(), 0.0)                   # no linear part
    assert np.allclose(bq, budget)
    assert np.allclose(prob.var_lb, 0.0)


def test_quadform_equals_xQx():
    rng = np.random.default_rng(1)
    M = rng.standard_normal((4, 4))
    x = dsl.Variable(4)
    fs = {x: slice(0, 4)}
    q1 = dsl.quadForm(x, M).finalize_Q(fs, 4)
    q2 = (x @ M @ x).finalize_Q(fs, 4)
    sym = (M + M.T) / 2
    assert np.allclose(q1.toarray(), sym)
    assert np.allclose(q2.toarray(), sym)


def test_qp_objective_offset():
    rng = np.random.default_rng(2)
    Sig = rng.standard_normal((3, 3))
    Sig = Sig @ Sig.T                                        # PSD
    x = dsl.Variable(3)
    c = dsl.Parameter(3)
    prob = dsl.Problem(dsl.Minimize(c @ x + dsl.quadForm(x, Sig)), [x.sum() == 1])
    assert prob.obj_Q is not None
    assert np.allclose(prob.obj_Q.toarray(), Sig)            # already symmetric


# ============================================================
# Cost layout: symbolic gather (one predicted edge cost -> directed arcs)
# ============================================================

def test_tsp_gather_records_edge_to_arc():
    # a 2-D arc->edge index must flatten to cost_gather in C-order (backend builds _cost_vars)
    n, edges = 3, [(0, 1), (0, 2), (1, 2)]
    emap = {e: i for i, e in enumerate(edges)}
    a2e = np.array([[0 if i == j else emap[tuple(sorted((i, j)))]
                     for j in range(n)] for i in range(n)])
    x = dsl.Variable((n, n), binary=True)
    c = dsl.Parameter(len(edges))
    prob = dsl.Problem(dsl.Minimize(dsl.sum(c[a2e] * x)), [x.sum() == 1])
    assert prob.num_cost == 3                        # predicted dimension = #edges
    assert prob.cost_var is x and prob.cost_var.size == 9
    assert np.array_equal(prob.cost_gather, a2e.reshape(-1))


def test_unused_cost_entry_keeps_num_cost():
    # Parameter(4) with only indices 0..2 used -> num_cost stays the declared size
    x = dsl.Variable(3)
    c = dsl.Parameter(4)
    prob = dsl.Problem(dsl.Minimize(dsl.sum(c[[0, 1, 2]] * x)), [x.sum() == 1])
    assert prob.num_cost == 4
    assert np.array_equal(prob.cost_gather, np.array([0, 1, 2]))


# ============================================================
# relax
# ============================================================

def test_relax_clears_vtypes_keeps_bounds():
    x = dsl.Variable(4, binary=True)
    c = dsl.Parameter(4)
    prob = dsl.Problem(dsl.Maximize(c @ x), [x.sum() <= 2])
    rel = prob.relax()
    assert not rel.var_binary.any() and not rel.var_integer.any()
    assert np.allclose(rel.var_lb, 0.0) and np.allclose(rel.var_ub, 1.0)
    assert prob.var_binary.all()                            # original unchanged


# ============================================================
# Validation
# ============================================================

def test_objective_must_be_parametric():
    x = dsl.Variable(3)
    with pytest.raises(TypeError):
        dsl.Minimize(x.sum())                               # no Parameter


def test_parameter_cannot_form_constraint():
    x = dsl.Variable(3)
    c = dsl.Parameter(3)
    with pytest.raises(TypeError):
        _ = (c @ x <= 1.0)                                  # ParametricBilinear has no <=


def test_two_parameters_rejected():
    x = dsl.Variable(3)
    y = dsl.Variable(3)
    c1 = dsl.Parameter(3)
    c2 = dsl.Parameter(3)
    with pytest.raises(TypeError):
        _ = c1 @ x + c2 @ y                                 # two ParametricBilinear


def test_parameter_size_mismatch_rejected():
    x = dsl.Variable(4)
    c = dsl.Parameter(3)
    with pytest.raises(TypeError):
        _ = c @ x                                           # 3 != 4, no gather


def test_parameter_forbidden_ops():
    c = dsl.Parameter(3)
    with pytest.raises(TypeError):
        _ = c + 1
