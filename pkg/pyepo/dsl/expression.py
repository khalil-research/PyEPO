#!/usr/bin/env python
"""
Symbolic expression nodes and algebra for the PyEPO DSL.

Variables and Parameters compose through numpy-style operators into linear
(``Affine``) and quadratic (``Quadratic``) nodes; comparisons produce a
``Constraint``; a single ``Parameter`` multiplies a Variable (or slice) into a
``ParametricObjective`` objective. Each linear node carries per-Variable
coefficient blocks (``dict[Variable, csr_matrix]``) plus a constant offset;
``Problem`` finalizes them into one global sparse matrix (see
``pyepo.dsl.problem``).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from pyepo import EPO


def _is_num(x) -> bool:
    # scalar or ndarray constant (anything that is not a DSL node)
    return isinstance(x, (int, float, np.ndarray, np.generic))


class Variable:
    """
    A decision variable of a given shape and per-entry type.

    ``vtype`` is a scalar ``EPO.VarType`` (uniform) or a per-entry array; binary
    entries are forced to bounds ``[0, 1]``.

    Attributes:
        shape (tuple): logical shape
        size (int): number of scalar entries (``prod(shape)``)
        vtype (np.ndarray): per-entry ``EPO.VarType``, length ``size``
        lb / ub (np.ndarray): flat C-order bounds, length ``size``
    """

    def __init__(self, shape, *, vtype=EPO.CONTINUOUS, lb=None, ub=None, name=None):
        self.shape = tuple(int(s) for s in np.atleast_1d(shape))
        self.size = int(np.prod(self.shape)) if self.shape else 1
        # per-entry type: a scalar EPO.VarType broadcast, or a per-entry array
        self.vtype = np.broadcast_to(np.asarray(vtype, dtype=object), self.shape).reshape(-1).copy()
        if not all(isinstance(t, EPO.VarType) for t in self.vtype):
            raise ValueError("vtype entries must be EPO.VarType (BINARY / INTEGER / CONTINUOUS).")
        # bounds (binary entries forced to [0, 1])
        is_bin = self.vtype == EPO.BINARY
        self.lb = np.where(is_bin, 0.0, self._bound(lb, -np.inf))
        self.ub = np.where(is_bin, 1.0, self._bound(ub, np.inf))
        self.name = name

    __hash__ = object.__hash__  # identity hash (kept after __eq__ override)
    __array_ufunc__ = None  # numpy defers @ / * / + to our reflected ops

    def _bound(self, val, default):
        # broadcast a scalar / array bound to a flat (size,) vector
        if val is None:
            val = default
        return np.broadcast_to(np.asarray(val, dtype=float), self.shape).reshape(-1).astype(float)

    def _to_affine(self) -> Affine:
        # identity affine: this variable as a linear expression
        return Affine({self: sp.eye(self.size, format="csr")}, np.zeros(self.size), self.shape)

    # ---- algebra: delegate to the affine view ----
    def __add__(self, o):
        return self._to_affine() + o

    def __radd__(self, o):
        return self._to_affine() + o

    def __sub__(self, o):
        return self._to_affine() - o

    def __rsub__(self, o):
        return (-self._to_affine()) + o

    def __neg__(self):
        return -self._to_affine()

    def __mul__(self, o):
        return self._to_affine() * o

    def __rmul__(self, o):
        return self._to_affine() * o

    def __matmul__(self, o):
        return self._to_affine() @ o

    def __rmatmul__(self, o):
        return o @ self._to_affine()

    def __getitem__(self, idx):
        return self._to_affine()[idx]

    def sum(self, axis=None):
        return self._to_affine().sum(axis)

    # ---- constraints ----
    def __le__(self, rhs):
        return self._to_affine() <= rhs

    def __ge__(self, rhs):
        return self._to_affine() >= rhs

    def __eq__(self, rhs):
        return self._to_affine() == rhs

    def __repr__(self):
        return f"Variable(shape={self.shape}, name={self.name!r})"


class Affine:
    """
    Linear expression ``Σ_v A_v @ vec(v) + b`` of logical ``shape``.

    Attributes:
        blocks (dict[Variable, csr_matrix]): per-Variable coefficient block ``(size, v.size)``
        const (np.ndarray): constant offset ``(size,)``
        shape (tuple): logical shape; ``size == prod(shape)``
    """

    def __init__(self, blocks, const, shape):
        self.shape = tuple(shape)
        self.size = int(np.prod(self.shape)) if self.shape else 1
        self.const = np.broadcast_to(np.asarray(const, dtype=float), (self.size,)).copy()
        # store each block as csr (size, v.size)
        self.blocks = {v: sp.csr_matrix(b) for v, b in blocks.items()}

    __hash__ = object.__hash__
    __array_ufunc__ = None  # numpy defers @ / * / + to our reflected ops

    def _add_block(self, blocks, v, b):
        # accumulate a coefficient block for variable v
        blocks[v] = blocks[v] + b if v in blocks else sp.csr_matrix(b)

    def __add__(self, o):
        # Affine + (Affine | Quadratic | ParametricObjective | const)
        if isinstance(o, Quadratic):
            return o + self
        if isinstance(o, ParametricObjective):
            return o + self
        if isinstance(o, Variable):
            o = o._to_affine()
        if isinstance(o, Affine):
            blocks = {v: b.copy() for v, b in self.blocks.items()}
            for v, b in o.blocks.items():
                self._add_block(blocks, v, b)
            return Affine(blocks, self.const + o.const, self.shape)
        if _is_num(o):
            return Affine(
                self.blocks, self.const + np.asarray(o, dtype=float).reshape(-1), self.shape
            )
        return NotImplemented

    def __radd__(self, o):
        # reflected add
        return self + o

    def __neg__(self):
        # negate every coefficient block and the constant
        return Affine({v: -b for v, b in self.blocks.items()}, -self.const, self.shape)

    def __sub__(self, o):
        # subtract
        if isinstance(o, (Affine, Variable, Quadratic)):
            return self + (-o)
        if _is_num(o):
            return self + (-np.asarray(o, dtype=float))
        return NotImplemented

    def __rsub__(self, o):
        # reflected subtract
        return (-self) + o

    def __mul__(self, o):
        # elementwise scale by a scalar or shape-broadcast array
        if isinstance(o, (Affine, Variable, Quadratic, Parameter, ParametricObjective)):
            return NotImplemented  # expr*expr is nonlinear; use @ instead
        # diagonal scaling matrix applied to each block and the constant
        a = np.broadcast_to(np.asarray(o, dtype=float), self.shape).reshape(-1)
        scale = sp.diags(a)
        return Affine({v: scale @ b for v, b in self.blocks.items()}, a * self.const, self.shape)

    def __rmul__(self, o):
        # reflected scale
        return self * o

    def __matmul__(self, o):
        # Affine @ (ndarray)  -> Affine ;  Affine @ (Variable | Affine) -> Quadratic
        if isinstance(o, Variable):
            o = o._to_affine()
        if isinstance(o, Affine):
            return _bilinear(self, o)
        if _is_num(o):
            # affine(k,) @ M(k,p): result block = M.T @ A_block ; const = M.T @ b
            M = np.asarray(o, dtype=float)
            blocks = {v: sp.csr_matrix(M.T @ b) for v, b in self.blocks.items()}
            const = M.T @ self.const
            return Affine(blocks, const, (M.shape[1],) if M.ndim == 2 else ())
        return NotImplemented

    def __rmatmul__(self, o):
        # ndarray @ Affine : M(p,k) @ affine(k,) -> Affine(p,)
        M = np.asarray(o, dtype=float)
        blocks = {v: sp.csr_matrix(M @ b) for v, b in self.blocks.items()}
        const = M @ self.const
        return Affine(blocks, const, (M.shape[0],) if M.ndim == 2 else ())

    def _row_op(self, S, shape):
        # apply a row-selection / reduction matrix S (p, m) -> new Affine
        blocks = {v: sp.csr_matrix(S @ b) for v, b in self.blocks.items()}
        return Affine(blocks, S @ self.const, shape)

    def sum(self, axis=None):
        # reduce along axis (or fully) via a 0/1 summation matrix
        if axis is None:
            return self._row_op(np.ones((1, self.size)), ())
        idx = np.arange(self.size).reshape(self.shape)
        kept = tuple(d for a, d in enumerate(self.shape) if a != axis)
        out_m = int(np.prod(kept)) if kept else 1
        # build summation matrix: out row r sums the flat cols collapsed into it
        keep_idx = np.moveaxis(idx, axis, -1).reshape(out_m, self.shape[axis])
        S = sp.lil_matrix((out_m, self.size))
        for r in range(out_m):
            S[r, keep_idx[r]] = 1.0
        return self._row_op(S.tocsr(), kept)

    def __getitem__(self, idx):
        # numpy-style indexing -> row selection
        sel = np.arange(self.size).reshape(self.shape)[idx]
        out_shape = sel.shape if hasattr(sel, "shape") else ()
        sel = np.atleast_1d(sel).reshape(-1)
        S = sp.csr_matrix(
            (np.ones(sel.size), (np.arange(sel.size), sel)), shape=(sel.size, self.size)
        )
        return self._row_op(S, out_shape)

    # ---- constraints ----
    def __le__(self, rhs):
        return Constraint(self, "<=", rhs)

    def __ge__(self, rhs):
        return Constraint(self, ">=", rhs)

    def __eq__(self, rhs):
        return Constraint(self, "==", rhs)

    def finalize(self, flat_slice, num_vars):
        # scatter each variable's block into the global matrix (size, num_vars)
        A = sp.lil_matrix((self.size, num_vars))
        for v, b in self.blocks.items():
            A[:, flat_slice[v]] = b
        return A.tocsr()


def _bilinear(e1: Affine, e2: Affine) -> Quadratic:
    # inner product (A1 x + b1)·(A2 x + b2) of two equal-length affines -> scalar Quadratic
    if e1.size != e2.size:
        raise ValueError(f"Inner product requires equal lengths, got {e1.size} and {e2.size}.")
    quad = {}
    for vi, Ai in e1.blocks.items():
        for vj, Aj in e2.blocks.items():
            quad[(vi, vj)] = quad.get((vi, vj), 0) + Ai.T @ Aj
    # linear part: A1^T b2 + A2^T b1 ; constant: b1·b2
    lin = {}
    for vi, Ai in e1.blocks.items():
        lin[vi] = lin.get(vi, 0) + sp.csr_matrix(Ai.T @ e2.const.reshape(-1, 1)).T
    for vj, Aj in e2.blocks.items():
        lin[vj] = lin.get(vj, 0) + sp.csr_matrix(Aj.T @ e1.const.reshape(-1, 1)).T
    affine = Affine(lin, np.array([float(e1.const @ e2.const)]), ())
    return Quadratic(quad, affine)


class Quadratic:
    """
    Scalar quadratic expression ``xᵀ Q x + (linear) + const``.

    Attributes:
        quad (dict[(Variable, Variable), csr_matrix]): symmetric-when-finalized blocks
        affine (Affine): scalar linear + constant part
    """

    def __init__(self, quad, affine):
        self.quad = quad
        self.affine = affine

    __hash__ = object.__hash__
    __array_ufunc__ = None  # numpy defers @ / * / + to our reflected ops

    def __add__(self, o):
        # Quadratic + (Affine | Variable | Quadratic | const) -> Quadratic
        if isinstance(o, Variable):
            o = o._to_affine()
        # affine / constant folds into the linear part
        if isinstance(o, Affine):
            return Quadratic(self.quad, self.affine + o)
        # merge the two quadratic block dicts
        if isinstance(o, Quadratic):
            quad = dict(self.quad)
            for k, b in o.quad.items():
                quad[k] = quad.get(k, 0) + b
            return Quadratic(quad, self.affine + o.affine)
        if _is_num(o):
            return Quadratic(self.quad, self.affine + o)
        return NotImplemented

    def __radd__(self, o):
        # reflected add
        return self + o

    def __mul__(self, o):
        # scale the quadratic and its affine part by a scalar
        if _is_num(o):
            s = float(o)
            return Quadratic({k: s * b for k, b in self.quad.items()}, self.affine * s)
        return NotImplemented

    def __rmul__(self, o):
        # reflected scale
        return self * o

    def __neg__(self):
        # negate the quadratic and its affine part
        return self * -1.0

    # ---- quadratic constraints ----
    def __le__(self, rhs):
        return Constraint(self, "<=", rhs)

    def __ge__(self, rhs):
        return Constraint(self, ">=", rhs)

    def __eq__(self, rhs):
        return Constraint(self, "==", rhs)

    def finalize_Q(self, flat_slice, num_vars):
        # scatter each block into the global matrix, then symmetrize
        Q = sp.lil_matrix((num_vars, num_vars))
        for (vi, vj), b in self.quad.items():
            Q[flat_slice[vi], flat_slice[vj]] += b.toarray() if sp.issparse(b) else b
        Q = Q.tocsr()
        return (Q + Q.T) * 0.5


def _is_row_selection(S, rows):
    # one entry of value 1 per row, distinct columns (a plain index / slice pick)
    return (
        S.nnz == rows
        and (S.data == 1).all()
        and np.array_equal(S.indptr, np.arange(rows + 1))
        and len(set(S.indices)) == S.nnz
    )


def _as_selection(operand):
    # a Variable (the whole thing), or a plain slice / index of one Variable
    if isinstance(operand, Variable):
        return operand, None
    if isinstance(operand, Affine) and len(operand.blocks) == 1 and not operand.const.any():
        ((var, S),) = operand.blocks.items()
        S = S.tocsr()
        if _is_row_selection(S, operand.size):
            return var, S.indices.copy()  # selected columns of var
    raise TypeError(
        "A Parameter may only multiply a Variable or a plain slice / index "
        "of one (e.g. x, x[:k], x[idx])."
    )


def _make_objective(param, base, sel, inner=False):
    # pair a predicted cost (param, optional known base) with a Variable / selection
    var, local_idx = _as_selection(sel)
    # `@` is a 1-D inner product; a multi-dimensional cost must go through (c * x).sum()
    if inner and len(getattr(sel, "shape", ())) > 1:
        raise TypeError(
            "c @ x is a 1-D inner product; for a multi-dimensional cost write (c * x).sum()."
        )
    n = var.size if local_idx is None else len(local_idx)
    if param.size != n:
        raise TypeError(
            f"Parameter size {param.size} does not match the selected variable size {n}."
        )
    return ParametricObjective(param, var, local_idx, base=base)


class Parameter:
    """
    The runtime-bound predicted cost symbol (exactly one per Problem).

    The inner product ``c @ x`` (or ``c @ x[:k]``) pairs the predicted cost with a
    Variable (or a plain slice / index of one) into a scalar objective term. The
    elementwise product ``c * x`` is a vector of predicted terms; reduce it to the
    scalar objective with ``(c * x).sum()``. A known base may be added first:
    ``(d + c) @ x`` predicts ``d + c`` on x. Any other use raises ``TypeError``.

    Attributes:
        shape (tuple): logical shape
        size (int): number of predicted entries (``prod(shape)``)
    """

    def __init__(self, shape, *, name=None):
        self.shape = tuple(int(s) for s in np.atleast_1d(shape))
        self.size = int(np.prod(self.shape)) if self.shape else 1
        self.name = name

    __hash__ = object.__hash__
    __array_ufunc__ = None  # numpy defers @ / * / + to our reflected ops

    def _inner(self, sel):
        # c @ x : inner product -> scalar predicted objective term
        return _make_objective(self, None, sel, inner=True)

    def _elem(self, sel):
        # c * x : elementwise -> vector of predicted terms (reduce with .sum())
        return ParametricVector(self, None, sel)

    __matmul__ = __rmatmul__ = _inner
    __mul__ = __rmul__ = _elem

    def __add__(self, base):
        # (c + d): a predicted cost with a known base
        if _is_num(base):
            return ParametricCoef(self, base)
        return self._forbid()

    def __radd__(self, base):
        return self.__add__(base)

    def __sub__(self, base):
        if _is_num(base):
            return ParametricCoef(self, -np.asarray(base, dtype=float))
        return self._forbid()

    def _forbid(self, *a, **k):
        raise TypeError(
            "Unsupported operation on Parameter (only c @ var / c * var, "
            "optionally with a known base: (d + c) @ var)."
        )

    __rsub__ = __neg__ = __getitem__ = __le__ = __ge__ = __eq__ = _forbid

    def __repr__(self):
        return f"Parameter(shape={self.shape}, name={self.name!r})"


class ParametricCoef:
    """
    A predicted cost coefficient with a known base, ``base + param`` — the
    transient result of ``c + d`` / ``d + c`` / ``c - d``. Pairs with a Variable
    or selection like a Parameter: ``(d + c) @ x`` (scalar) or ``(d + c) * x``
    (elementwise vector, reduce with ``.sum()``).
    """

    def __init__(self, param, base):
        self.param = param
        self.base = np.broadcast_to(np.asarray(base, dtype=float), (param.size,)).copy()

    __array_ufunc__ = None

    def _inner(self, sel):
        # (d + c) @ x : inner product -> scalar predicted objective term
        return _make_objective(self.param, self.base, sel, inner=True)

    def _elem(self, sel):
        # (d + c) * x : elementwise -> vector of predicted terms (reduce with .sum())
        return ParametricVector(self.param, self.base, sel)

    __matmul__ = __rmatmul__ = _inner
    __mul__ = __rmul__ = _elem


class ParametricVector:
    """
    The elementwise product ``c * x`` (optionally with a known base): a vector of
    predicted terms, one per selected position. It is not yet a scalar objective;
    reduce it with ``.sum()`` to form the objective term ``(c * x).sum()``.

    Attributes:
        param (Parameter): the predicted cost
        base (np.ndarray | None): known offset added at the predicted positions
        sel (Variable | Affine): the Variable / selection the cost lands on
    """

    def __init__(self, param, base, sel):
        # validate the pairing eagerly so size errors fire at `c * x`, not at `.sum()`
        var, local_idx = _as_selection(sel)
        n = var.size if local_idx is None else len(local_idx)
        if param.size != n:
            raise TypeError(
                f"Parameter size {param.size} does not match the selected variable size {n}."
            )
        self.param = param
        self.base = base
        self.sel = sel

    __hash__ = object.__hash__
    __array_ufunc__ = None  # numpy defers @ / * / + to our reflected ops

    def sum(self, axis=None):
        # reduce to the scalar predicted objective term
        if axis is not None:
            raise TypeError(
                "A predicted objective must reduce to a scalar; use .sum() with no axis."
            )
        return _make_objective(self.param, self.base, self.sel)

    def __add__(self, o):
        raise TypeError(
            "c * x is an elementwise vector; reduce it with .sum() before adding "
            "other terms, e.g. (c * x).sum() + d @ y."
        )

    __radd__ = __add__


class ParametricObjective:
    """
    The objective: ``cost_param`` predicts the cost at ``cost_var``'s selected
    positions (with an optional known ``base``), plus optional known fixed
    linear (``fixed``) and parameter-free quadratic (``quad``) terms.

    Attributes:
        cost_param (Parameter): the predicted cost (size == num_cost)
        cost_var (Variable): the Variable the prediction lands on
        local_idx (np.ndarray | None): predicted indices within cost_var (None = all)
        base (np.ndarray | None): known cost added at the predicted positions
        fixed (Affine | None): known linear term on other / overlapping positions
        quad (Quadratic | None): parameter-free quadratic term
    """

    def __init__(self, cost_param, cost_var, local_idx=None, base=None, fixed=None, quad=None):
        self.cost_param = cost_param
        self.cost_var = cost_var
        self.local_idx = local_idx
        self.base = base
        self.fixed = fixed
        self.quad = quad

    __hash__ = object.__hash__
    __array_ufunc__ = None  # numpy defers @ / * / + to our reflected ops

    def _with(self, fixed=None, quad=None):
        # copy with a replaced fixed / quad term
        return ParametricObjective(
            self.cost_param,
            self.cost_var,
            self.local_idx,
            self.base,
            self.fixed if fixed is None else fixed,
            self.quad if quad is None else quad,
        )

    def sum(self, axis=None):
        # already a scalar; sum is the identity here
        return self

    def __add__(self, o):
        # attach a known linear (Affine / Variable) or quadratic (Quadratic) term
        if isinstance(o, ParametricObjective):
            raise TypeError(
                "Cannot add two predicted cost terms; a problem has exactly one predicted cost."
            )
        if isinstance(o, Quadratic):
            return self._with(quad=o if self.quad is None else self.quad + o)
        if isinstance(o, Variable):
            o = o._to_affine()
        if isinstance(o, Affine):
            return self._with(fixed=o if self.fixed is None else self.fixed + o)
        if _is_num(o):
            raise TypeError("A constant objective term has no effect; omit it.")
        return NotImplemented

    def __radd__(self, o):
        return self + o


class Constraint:
    """
    A relation ``lhs (<= | >= | ==) rhs`` between an expression and a constant.

    Attributes:
        lhs (Affine | Quadratic): left-hand side expression
        sense (str): one of ``"<="`` / ``">="`` / ``"=="``
        rhs (np.ndarray): right-hand side, broadcast to the LHS length
    """

    def __init__(self, lhs, sense, rhs):
        # rhs must be a numeric constant, not a DSL expression
        try:
            np.asarray(rhs, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("Constraint rhs must be a constant (scalar or array).") from exc
        self.lhs = lhs
        self.sense = sense
        self.rhs = rhs

    def __bool__(self):
        # guard against accidental truth-testing (e.g. `if expr == rhs`)
        raise TypeError("A Constraint is not a boolean; use it inside Problem(constraints=[...]).")

    def variables(self):
        # Variables referenced by the LHS (encounter order); dedup by identity
        if isinstance(self.lhs, Quadratic):
            order, seen = [], set()
            cand = list(self.lhs.affine.blocks)
            for vi, vj in self.lhs.quad:
                cand.extend((vi, vj))
            for v in cand:
                if id(v) not in seen:
                    seen.add(id(v))
                    order.append(v)
            return order
        return list(self.lhs.blocks)

    def finalize(self, flat_slice, num_vars):
        """
        Assemble the global form ``(Q | None, A, sense, b_eff)`` (``b_eff = rhs - const``).
        """
        # quadratic constraint: keep Q and the linear part, fold const into rhs
        if isinstance(self.lhs, Quadratic):
            aff = self.lhs.affine
            Q = self.lhs.finalize_Q(flat_slice, num_vars)
            A = aff.finalize(flat_slice, num_vars)
            return Q, A, self.sense, self._rhs_vec(aff.size) - aff.const
        # linear constraint: no Q, fold const into rhs
        aff = self.lhs
        A = aff.finalize(flat_slice, num_vars)
        return None, A, self.sense, self._rhs_vec(aff.size) - aff.const

    def _rhs_vec(self, m):
        # broadcast the rhs to a flat (m,) vector
        return np.broadcast_to(np.asarray(self.rhs, dtype=float), (m,)).astype(float)
