#!/usr/bin/env python
"""Tests for pyepo.dsl: the symbolic IR core and the solver backends.

The IR tests are solver-free (pure numpy/scipy): the finalized sparse matrices
match hand-derived forms for the canonical shapes (knapsack, shortest path,
portfolio, assignment), the operators are correct, and scope validation fires.
The backend tests are parametrized over Gurobi / COPT (each skipped when its
solver is absent) and check golden equivalence to the legacy hand-written
classes plus solve properties.
"""

import numpy as np
import pytest

from pyepo import EPO, dsl
from pyepo.model.opt import optModel

from .conftest import requires_copt, requires_gurobi

# ============================================================
# Variable / Parameter construction
# ============================================================

def test_variable_binary_bounds():
    x = dsl.Variable(4, vtype=EPO.BINARY)
    assert x.size == 4 and (x.vtype == EPO.BINARY).all()
    assert np.allclose(x.lb, 0) and np.allclose(x.ub, 1)


def test_variable_default_is_continuous():
    x = dsl.Variable(3)
    assert (x.vtype == EPO.CONTINUOUS).all()


def test_variable_invalid_vtype_rejected():
    with pytest.raises(ValueError):
        dsl.Variable(3, vtype="B")                          # must be EPO.VarType, not a raw code


def test_variable_multidim_and_bounds_broadcast():
    x = dsl.Variable((2, 3), lb=-1.0, ub=2.0)
    assert x.shape == (2, 3) and x.size == 6
    assert np.allclose(x.lb, -1.0) and np.allclose(x.ub, 2.0)


def test_variable_mixed_vtype_per_entry():
    x = dsl.Variable(4, vtype=[EPO.BINARY, EPO.BINARY, EPO.INTEGER, EPO.CONTINUOUS], lb=0, ub=7)
    assert list(x.vtype) == [EPO.BINARY, EPO.BINARY, EPO.INTEGER, EPO.CONTINUOUS]
    assert np.allclose(x.ub, [1, 1, 7, 7])                 # binary entries forced to [0, 1]


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
    con = (x.sum() + 2.0 == 5.0)            # const folds into rhs (5 - 2)
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
    x = dsl.Variable(5, vtype=EPO.BINARY)
    c = dsl.Parameter(5)
    prob = dsl.Problem(dsl.Maximize(c @ x), [W @ x <= cap])
    assert prob.num_vars == 5 and prob.num_cost == 5
    assert prob.objective.modelSense == EPO.MAXIMIZE
    Q, A, sense, b = prob.constrs[0]
    assert Q is None and sense == "<="
    assert np.allclose(A.toarray(), W) and np.allclose(b, cap)
    assert (prob.var_type == EPO.BINARY).all()


def test_shortestpath_equality_ir():
    # generic flow-conservation equality A_eq x = b
    A_eq = np.array([[1.0, 1, 0, 0], [-1, 0, 1, 0], [0, -1, -1, 1]])
    b_eq = np.array([1.0, 0, -1])
    x = dsl.Variable(4)
    c = dsl.Parameter(4)
    prob = dsl.Problem(dsl.Minimize(c @ x), [A_eq @ x == b_eq])
    _, A, sense, b = prob.constrs[0]
    assert sense == "==" and np.allclose(A.toarray(), A_eq) and np.allclose(b, b_eq)
    assert prob.objective.modelSense == EPO.MINIMIZE


def test_assignment_ir():
    x = dsl.Variable((3, 3), vtype=EPO.BINARY)
    c = dsl.Parameter((3, 3))
    prob = dsl.Problem(dsl.Minimize(dsl.sum(c * x)),
                       [x.sum(axis=1) == 1, x.sum(axis=0) == 1])
    assert prob.num_cost == 9
    _, Ar, _, _ = prob.constrs[0]
    _, Ac, _, _ = prob.constrs[1]
    assert np.allclose(Ar.toarray(), np.kron(np.eye(3), np.ones((1, 3))))
    assert np.allclose(Ac.toarray(), np.kron(np.ones((1, 3)), np.eye(3)))


def test_portfolio_quadratic_constraint_ir():
    rng = np.random.default_rng(0)
    cov = np.cov(rng.standard_normal((50, 5)), rowvar=False)
    budget = 2.25 * cov.mean()
    x = dsl.Variable(5, lb=0)
    c = dsl.Parameter(5)
    prob = dsl.Problem(dsl.Maximize(c @ x),
                       [x.sum() == 1, x @ cov @ x <= budget])
    _, As, _, bs = prob.constrs[0]
    Qq, Aq, sq, bq = prob.constrs[1]
    assert np.allclose(As.toarray(), np.ones((1, 5))) and np.allclose(bs, 1.0)
    assert sq == "<=" and Qq is not None
    assert np.allclose(Qq.toarray(), (cov + cov.T) / 2)     # symmetric quadratic part
    assert np.allclose(Aq.toarray(), 0.0)                   # no linear part
    assert np.allclose(bq, budget)
    assert np.allclose(prob.var_lb, 0.0)


def test_quadratic_equality_constraint():
    x = dsl.Variable(2)
    c = dsl.Parameter(2)
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    prob = dsl.Problem(dsl.Minimize(c @ x), [x @ cov @ x == 1.0])
    Q, _, sense, b = prob.constrs[0]                        # `==` builds a real Constraint
    assert sense == "==" and Q is not None
    assert np.allclose(Q.toarray(), cov) and np.allclose(b, 1.0)


def test_xQx_finalizes_symmetric():
    rng = np.random.default_rng(1)
    M = rng.standard_normal((4, 4))
    x = dsl.Variable(4)
    q = (x @ M @ x).finalize_Q({x: slice(0, 4)}, 4)
    assert np.allclose(q.toarray(), (M + M.T) / 2)          # x @ M @ x symmetrizes M


def test_quadratic_scale_and_add():
    x = dsl.Variable(2)
    A = np.eye(2)
    B = np.array([[0.0, 1.0], [1.0, 0.0]])
    Q = (0.5 * (x @ A @ x) + x @ B @ x).finalize_Q({x: slice(0, 2)}, 2)
    assert np.allclose(Q.toarray(), 0.5 * A + B)            # scalar scale + merge


def test_qp_objective_offset():
    rng = np.random.default_rng(2)
    Sig = rng.standard_normal((3, 3))
    Sig = Sig @ Sig.T                                        # PSD
    x = dsl.Variable(3)
    c = dsl.Parameter(3)
    prob = dsl.Problem(dsl.Minimize(c @ x + x @ Sig @ x), [x.sum() == 1])
    assert prob.obj_Q is not None
    assert np.allclose(prob.obj_Q.toarray(), Sig)            # already symmetric


# ============================================================
# relax
# ============================================================

def test_relax_clears_vtypes_keeps_bounds():
    x = dsl.Variable(4, vtype=EPO.BINARY)
    c = dsl.Parameter(4)
    prob = dsl.Problem(dsl.Maximize(c @ x), [x.sum() <= 2])
    rel = prob.relax()
    assert (rel.var_type == EPO.CONTINUOUS).all()
    assert np.allclose(rel.var_lb, 0.0) and np.allclose(rel.var_ub, 1.0)
    assert (prob.var_type == EPO.BINARY).all()              # original unchanged


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
        _ = c @ x                                           # 3 != 4


def test_parameter_forbidden_ops():
    c = dsl.Parameter(3)
    for bad in (lambda: c == 5, lambda: c <= 5, lambda: -c, lambda: c[0], lambda: 5 - c):
        with pytest.raises(TypeError):
            bad()


def test_constraint_rhs_must_be_constant():
    x = dsl.Variable(3)
    c = dsl.Parameter(3)
    with pytest.raises(TypeError):
        _ = (x.sum() <= c)                                  # Parameter as rhs
    with pytest.raises(TypeError):
        _ = (x - c)                                         # Affine - Parameter


def test_objective_rejects_constant():
    x = dsl.Variable(3)
    c = dsl.Parameter(3)
    with pytest.raises(TypeError):
        _ = c @ x + 5.0                                     # a constant term has no effect


# ============================================================
# Partial prediction: known + predicted costs
# ============================================================

def test_partial_prediction_ir():
    x = dsl.Variable(3)
    y = dsl.Variable(2)
    c = dsl.Parameter(3)
    d = np.array([1.0, 2.0])
    prob = dsl.Problem(dsl.Minimize(c @ x + d @ y), [x.sum() + y.sum() == 1])
    assert prob.num_cost == 3 and prob.num_vars == 5
    assert list(prob.c_index) == [0, 1, 2]          # x predicted
    assert np.allclose(prob.fixed_c, [0, 0, 0, 1, 2])    # d fixed on y


def test_slice_prediction_ir():
    x = dsl.Variable(5)
    c = dsl.Parameter(2)
    d = np.array([3.0, 4.0, 5.0])
    prob = dsl.Problem(dsl.Minimize(c @ x[:2] + d @ x[2:]), [x.sum() == 1])
    assert list(prob.c_index) == [0, 1]             # first 2 of x predicted
    assert np.allclose(prob.fixed_c, [0, 0, 3, 4, 5])    # rest of x fixed


def test_base_offset_equals_two_terms():
    x = dsl.Variable(3)
    c = dsl.Parameter(3)
    d = np.array([1.0, 2.0, 3.0])
    combined = dsl.Problem(dsl.Minimize((d + c) @ x), [x.sum() == 1])      # (d + c) @ x
    two_term = dsl.Problem(dsl.Minimize(c @ x + d @ x), [x.sum() == 1])    # c @ x + d @ x
    assert np.allclose(combined.fixed_c, d) and np.allclose(two_term.fixed_c, d)
    assert list(combined.c_index) == list(two_term.c_index) == [0, 1, 2]


# ============================================================
# Backend golden equivalence + solve properties (parametrized over backend)
# ============================================================

_BACKENDS = [
    pytest.param("gurobi", marks=requires_gurobi),
    pytest.param("copt", marks=requires_copt),
]


def _legacy(backend):
    # the legacy hand-written model module for this backend
    if backend == "gurobi":
        import pyepo.model.grb as m
    else:
        import pyepo.model.copt as m
    return m


@pytest.mark.parametrize("backend", _BACKENDS)
def test_knapsack_matches_legacy(backend):
    W = np.array([[3.0, 4, 3, 6, 4], [4, 5, 2, 3, 5], [5, 4, 6, 2, 3]])
    cap = np.array([12.0, 10, 15])
    legacy = _legacy(backend).knapsackModel(W, cap)
    x = dsl.Variable(5, vtype=EPO.BINARY)
    c = dsl.Parameter(5)
    comp = dsl.Problem(dsl.Maximize(c @ x), [W @ x <= cap]).compile(backend=backend)
    assert isinstance(comp, optModel) and comp.num_cost == 5
    rng = np.random.default_rng(0)
    for _ in range(30):
        cost = rng.standard_normal(5)
        legacy.setObj(cost)
        _, obj_l = legacy.solve()
        comp.setObj(cost)
        _, obj_d = comp.solve()
        assert obj_d == pytest.approx(obj_l, abs=1e-6)       # golden objective


@pytest.mark.parametrize("backend", _BACKENDS)
def test_relax_matches_legacy_lp(backend):
    W = np.array([[3.0, 4, 3, 6, 4], [4, 5, 2, 3, 5]])
    cap = np.array([9.0, 8.0])
    legacy_lp = _legacy(backend).knapsackModel(W, cap).relax()
    x = dsl.Variable(5, vtype=EPO.BINARY)
    c = dsl.Parameter(5)
    comp_lp = dsl.Problem(dsl.Maximize(c @ x), [W @ x <= cap]).compile(backend=backend).relax()
    assert (comp_lp.problem.var_type == EPO.CONTINUOUS).all()
    rng = np.random.default_rng(1)
    for _ in range(20):
        cost = rng.standard_normal(5)
        legacy_lp.setObj(cost)
        _, obj_l = legacy_lp.solve()
        comp_lp.setObj(cost)
        _, obj_d = comp_lp.solve()
        assert obj_d == pytest.approx(obj_l, abs=1e-6)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_portfolio_quadratic_constraint_solves(backend):
    rng = np.random.default_rng(2)
    cov = np.cov(rng.standard_normal((60, 5)), rowvar=False)
    budget = 2.25 * cov.mean()
    x = dsl.Variable(5, lb=0)
    c = dsl.Parameter(5)
    comp = dsl.Problem(dsl.Maximize(c @ x),
                       [x.sum() == 1, x @ cov @ x <= budget]).compile(backend=backend)
    comp.setObj(rng.standard_normal(5))
    sol, _ = comp.solve()
    assert sol.sum() == pytest.approx(1.0, abs=1e-5)        # budget constraint
    assert sol @ ((cov + cov.T) / 2) @ sol <= budget + 1e-6  # risk constraint
    assert (sol >= -1e-6).all()                             # lb = 0


@pytest.mark.parametrize("backend", _BACKENDS)
def test_assignment_solves_to_permutation(backend):
    n = 4
    x = dsl.Variable((n, n), vtype=EPO.BINARY)
    c = dsl.Parameter((n, n))
    comp = dsl.Problem(dsl.Minimize(dsl.sum(c * x)),
                       [x.sum(axis=1) == 1, x.sum(axis=0) == 1]).compile(backend=backend)
    rng = np.random.default_rng(3)
    comp.setObj(rng.standard_normal(n * n))
    sol, _ = comp.solve()
    P = np.asarray(sol).reshape(n, n)
    assert np.allclose(P.sum(axis=0), 1) and np.allclose(P.sum(axis=1), 1)   # permutation


@pytest.mark.parametrize("backend", _BACKENDS)
def test_qp_objective_solves(backend):
    # objective c @ x + xT Sig x
    Sig = np.array([[2.0, 0.5], [0.5, 1.0]])
    x = dsl.Variable(2, lb=0)
    c = dsl.Parameter(2)
    comp = dsl.Problem(dsl.Minimize(c @ x + x @ Sig @ x), [x.sum() == 1]).compile(backend=backend)
    comp.setObj(np.zeros(2))                                # zero linear cost: pure quadratic
    _, obj = comp.solve()
    grid = np.linspace(0, 1, 2001)
    brute = min(np.array([t, 1 - t]) @ Sig @ np.array([t, 1 - t]) for t in grid)
    assert obj == pytest.approx(brute, abs=1e-3)
    comp.setObj([1.0, 1.0])                                 # + linear term
    _, obj2 = comp.solve()
    assert obj2 == pytest.approx(obj + 1.0, abs=1e-3)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_mixed_vtype_solves(backend):
    # mixed-type cost variable
    x = dsl.Variable(3, vtype=[EPO.BINARY, EPO.INTEGER, EPO.CONTINUOUS], lb=0, ub=10)
    c = dsl.Parameter(3)
    comp = dsl.Problem(dsl.Maximize(c @ x), [x.sum() <= 4.5]).compile(backend=backend)
    comp.setObj([1.0, 1.0, 1.0])
    sol, _ = comp.solve()
    assert sol[0] in (0.0, 1.0)                             # binary entry
    assert abs(sol[1] - round(sol[1])) < 1e-6              # integer entry
    assert sol.sum() <= 4.5 + 1e-6


@pytest.mark.parametrize("backend", _BACKENDS)
def test_addconstr_cost_space(backend):
    W = np.array([[3.0, 4, 3, 6, 4]])
    cap = np.array([9.0])
    x = dsl.Variable(5, vtype=EPO.BINARY)
    c = dsl.Parameter(5)
    comp = dsl.Problem(dsl.Maximize(c @ x), [W @ x <= cap]).compile(backend=backend)
    comp2 = comp.addConstr([1, 1, 1, 0, 0], 1.0)           # cost-space cut (copy + add)
    comp2.setObj(np.ones(5))
    sol, _ = comp2.solve()
    assert np.asarray(sol)[:3].sum() <= 1 + 1e-6


@pytest.mark.parametrize("backend", _BACKENDS)
def test_partial_prediction_solves(backend):
    # predicted cost on x, known cost on y; y is expensive so x is used
    x = dsl.Variable(2, lb=0, ub=1)
    y = dsl.Variable(2, lb=0, ub=1)
    c = dsl.Parameter(2)
    d = np.array([5.0, 5.0])
    comp = dsl.Problem(dsl.Minimize(c @ x + d @ y),
                       [x[0] + y[0] >= 1, x[1] + y[1] >= 1]).compile(backend=backend)
    comp.setObj([1.0, 1.0])
    w, obj = comp.solve()
    assert len(w) == 2 and np.allclose(w, [1, 1])           # w is the predicted (x) part
    assert obj == pytest.approx(2.0)                        # c @ w; known d @ y removed


@pytest.mark.parametrize("backend", _BACKENDS)
def test_aux_variable_solves(backend):
    # y appears only in a constraint (no objective cost) -> not in w
    x = dsl.Variable(2, lb=0, ub=1)
    y = dsl.Variable(1, lb=0, ub=1.5)
    c = dsl.Parameter(2)
    comp = dsl.Problem(dsl.Maximize(c @ x), [x.sum() - y <= 0]).compile(backend=backend)
    comp.setObj([1.0, 1.0])
    w, obj = comp.solve()
    assert len(w) == 2 and obj == pytest.approx(1.5)        # x.sum() <= y <= 1.5


@pytest.mark.parametrize("backend", _BACKENDS)
def test_solver_params_silent_default(backend):
    x = dsl.Variable(3, vtype=EPO.BINARY)
    c = dsl.Parameter(3)
    comp = dsl.Problem(dsl.Maximize(c @ x), [np.ones((1, 3)) @ x <= 2]).compile(
        backend=backend, TimeLimit=12.5)

    def param(model, name):
        return getattr(model.Params, name) if backend == "gurobi" else model.getParam(name)

    silent = "OutputFlag" if backend == "gurobi" else "Logging"   # solver output flag
    assert param(comp._model, "TimeLimit") == 12.5         # user param applied
    assert param(comp._model, silent) == 0                 # silent by default
    assert param(comp.relax()._model, "TimeLimit") == 12.5  # params survive relax


@pytest.mark.parametrize("backend", _BACKENDS)
def test_constraints_named(backend):
    x = dsl.Variable(2, vtype=EPO.BINARY)
    c = dsl.Parameter(2)
    comp = dsl.Problem(dsl.Maximize(c @ x), [np.ones((1, 2)) @ x <= 1]).compile(backend=backend)
    comp._model.update()
    names = [con.ConstrName if backend == "gurobi" else con.name
             for con in comp._model.getConstrs()]
    assert any(n.startswith("c0") for n in names)          # constraints named c{i}
