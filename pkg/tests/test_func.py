#!/usr/bin/env python
"""
Tests for pyepo.func: utility functions and abstract module

Focuses on unit-level tests (solution pool, caching, helpers) without training.
"""

import numpy as np
import pytest
import torch

from pyepo import EPO
from pyepo.func.utils import _cache_in_pass, _check_sol, _update_solution_pool, sumGammaDistribution

try:
    from pyepo.model.grb.knapsack import knapsackModel
    from pyepo.model.grb.shortestpath import shortestPathModel
    # probe instantiation: import alone passes even without a valid license
    shortestPathModel(grid=(3, 3))
    _HAS_GUROBI = True
except Exception:
    _HAS_GUROBI = False

requires_gurobi = pytest.mark.skipif(not _HAS_GUROBI, reason="Gurobi not installed")


from pyepo.func.cave import coneAlignedCosine

# ============================================================
# _cache_in_pass tests
# ============================================================

class TestCacheInPass:

    def _make_minimize_model(self):
        from unittest.mock import MagicMock
        m = MagicMock()
        m.modelSense = EPO.MINIMIZE
        return m

    def _make_maximize_model(self):
        from unittest.mock import MagicMock
        m = MagicMock()
        m.modelSense = EPO.MAXIMIZE
        return m

    def test_minimize_selects_min_obj(self):
        model = self._make_minimize_model()
        # 2 instances, 3-dim cost
        cp = torch.tensor([[1.0, 2.0, 3.0],
                           [3.0, 2.0, 1.0]])
        # solution pool: 2 solutions
        solpool = torch.tensor([[1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0]])
        sol, _obj, _ = _cache_in_pass(cp, model, solpool)
        # instance 0: costs=[1,2,3], sol0 gives obj=1, sol1 gives obj=3 → pick sol0
        assert torch.allclose(sol[0], solpool[0])
        # instance 1: costs=[3,2,1], sol0 gives obj=3, sol1 gives obj=1 → pick sol1
        assert torch.allclose(sol[1], solpool[1])

    def test_maximize_selects_max_obj(self):
        model = self._make_maximize_model()
        cp = torch.tensor([[1.0, 2.0, 3.0],
                           [3.0, 2.0, 1.0]])
        solpool = torch.tensor([[1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0]])
        sol, _obj, _ = _cache_in_pass(cp, model, solpool)
        # instance 0: sol0→1, sol1→3 → pick sol1 (max)
        assert torch.allclose(sol[0], solpool[1])
        # instance 1: sol0→3, sol1→1 → pick sol0 (max)
        assert torch.allclose(sol[1], solpool[0])

    def test_invalid_sense_raises(self):
        from unittest.mock import MagicMock
        model = MagicMock()
        model.modelSense = 999
        cp = torch.tensor([[1.0, 2.0]])
        solpool = torch.tensor([[1.0, 0.0]])
        with pytest.raises(ValueError):
            _cache_in_pass(cp, model, solpool)

    def test_output_shapes(self):
        model = self._make_minimize_model()
        n, d, pool_size = 5, 4, 10
        cp = torch.randn(n, d)
        solpool = torch.randn(pool_size, d)
        sol, obj, _ = _cache_in_pass(cp, model, solpool)
        assert sol.shape == (n, d)
        assert obj.shape == (n,)


# ============================================================
# _check_sol tests
# ============================================================

class TestCheckSol:

    def test_correct_solution_passes(self):
        c = torch.tensor([[1.0, 2.0, 3.0]])
        w = torch.tensor([[1.0, 0.0, 1.0]])
        z = torch.tensor([4.0])  # 1*1 + 2*0 + 3*1 = 4
        # should not raise
        _check_sol(c, w, z)

    def test_incorrect_solution_raises(self):
        c = torch.tensor([[1.0, 2.0, 3.0]])
        w = torch.tensor([[1.0, 0.0, 1.0]])
        z = torch.tensor([999.0])  # wrong
        with pytest.raises(AssertionError):
            _check_sol(c, w, z)

    def test_batch(self):
        c = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        w = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        z = torch.tensor([3.0, 7.0])
        _check_sol(c, w, z)


# ============================================================
# sumGammaDistribution tests
# ============================================================

class TestSumGammaDistribution:

    def test_sample_shape(self):
        dist = sumGammaDistribution(kappa=1.0, n_iterations=10, seed=42)
        samples = dist.sample((5, 3))
        assert samples.shape == (5, 3)

    def test_deterministic(self):
        d1 = sumGammaDistribution(kappa=1.0, seed=42)
        d2 = sumGammaDistribution(kappa=1.0, seed=42)
        s1 = d1.sample((10,))
        s2 = d2.sample((10,))
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds(self):
        d1 = sumGammaDistribution(kappa=1.0, seed=0)
        d2 = sumGammaDistribution(kappa=1.0, seed=1)
        s1 = d1.sample((100,))
        s2 = d2.sample((100,))
        assert not np.array_equal(s1, s2)

    def test_finite_values(self):
        dist = sumGammaDistribution(kappa=2.0, n_iterations=20, seed=42)
        samples = dist.sample((100,))
        assert np.all(np.isfinite(samples))


# ============================================================
# Solution pool (_update_solution_pool) tests
# ============================================================

class TestSolutionPool:

    def test_init_pool(self):
        sol = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        solpool = _update_solution_pool(sol, None)
        assert solpool.shape == (2, 2)

    def test_dedup(self):
        sol1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        solpool = _update_solution_pool(sol1, None)
        # add same solutions again
        sol2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        solpool = _update_solution_pool(sol2, solpool)
        # pool should still have only 2 solutions
        assert solpool.shape[0] == 2

    def test_add_new(self):
        sol1 = torch.tensor([[1.0, 0.0]])
        solpool = _update_solution_pool(sol1, None)
        sol2 = torch.tensor([[0.0, 1.0]])
        solpool = _update_solution_pool(sol2, solpool)
        assert solpool.shape[0] == 2

    def test_mixed_new_and_dup(self):
        sol1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        solpool = _update_solution_pool(sol1, None)
        # one new, one duplicate
        sol2 = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        solpool = _update_solution_pool(sol2, solpool)
        assert solpool.shape[0] == 3

    def test_empty_new_after_dedup(self):
        sol = torch.tensor([[1.0, 2.0]])
        solpool = _update_solution_pool(sol, None)
        # all duplicates
        solpool = _update_solution_pool(sol, solpool)
        solpool = _update_solution_pool(sol, solpool)
        assert solpool.shape[0] == 1


# ============================================================
# optModule init validation tests
# ============================================================

@requires_gurobi
class TestOptModuleInit:

    def test_invalid_model_type_raises(self):
        from pyepo.func.surrogate import SPOPlus
        with pytest.raises(TypeError):
            SPOPlus("not_a_model")

    def test_invalid_processes_raises(self):
        from pyepo.func.surrogate import SPOPlus
        model = shortestPathModel(grid=(3, 3))
        with pytest.raises(ValueError):
            SPOPlus(model, processes=-1)

    def test_invalid_solve_ratio_raises(self):
        from pyepo.func.surrogate import SPOPlus
        model = shortestPathModel(grid=(3, 3))
        with pytest.raises(ValueError):
            SPOPlus(model, solve_ratio=1.5)
        with pytest.raises(ValueError):
            SPOPlus(model, solve_ratio=-0.1)

    def test_solve_ratio_lt1_requires_dataset(self):
        from pyepo.func.surrogate import SPOPlus
        model = shortestPathModel(grid=(3, 3))
        with pytest.raises(TypeError):
            SPOPlus(model, solve_ratio=0.5, dataset=None)


# ============================================================
# Autograd gradient correctness for SPO+
# ============================================================

@requires_gurobi
class TestSPOPlusGradient:
    """Check the SPO+ autograd subgradient against its closed-form expression."""

    def _setup(self, seed: int = 0):
        from pyepo.func.surrogate import SPOPlus
        rng = np.random.RandomState(seed)
        model = shortestPathModel(grid=(3, 3))
        n, d = 4, model.num_cost
        c_true = torch.as_tensor(rng.rand(n, d) + 0.1, dtype=torch.float32)
        # solve once to get w_true, z_true
        w_true = np.zeros((n, d), dtype=np.float32)
        z_true = np.zeros((n, 1), dtype=np.float32)
        for i in range(n):
            model.setObj(c_true[i].numpy())
            sol, obj = model.solve()
            w_true[i] = np.asarray(sol, dtype=np.float32)
            z_true[i, 0] = obj
        w_true = torch.from_numpy(w_true)
        z_true = torch.from_numpy(z_true)
        spo = SPOPlus(model, processes=1)
        return spo, model, c_true, w_true, z_true

    def test_subgradient_matches_closed_form(self):
        # subgradient = 2 * (w_true - argmin(2*cp - c_true)) / batch_size
        spo, model, c_true, w_true, z_true = self._setup(seed=0)
        cp = (c_true * 1.3).clone().detach().requires_grad_(True)
        loss = spo(cp, c_true, w_true, z_true)
        loss.backward()
        # closed-form subgradient
        spo_costs = 2 * cp.detach() - c_true
        w_spo = np.zeros_like(cp.detach().numpy())
        for i in range(cp.shape[0]):
            model.setObj(spo_costs[i].numpy())
            sol, _ = model.solve()
            w_spo[i] = np.asarray(sol, dtype=np.float32)
        expected = 2.0 * (w_true.numpy() - w_spo) / cp.shape[0]
        np.testing.assert_allclose(cp.grad.numpy(), expected, atol=1e-4)

    def test_gradient_step_decreases_loss(self):
        # one autograd step against the grad reduces SPO+ loss
        spo, _, c_true, w_true, z_true = self._setup(seed=1)
        cp = (c_true * 1.5).clone().detach().requires_grad_(True)
        loss0 = spo(cp, c_true, w_true, z_true)
        loss0.backward()
        with torch.no_grad():
            cp_new = cp - 0.1 * cp.grad
        loss1 = spo(cp_new, c_true, w_true, z_true)
        assert loss1.item() <= loss0.item() + 1e-6


# ============================================================
# Frank-Wolfe inner loop (used by regularizedFrankWolfeOpt)
# ============================================================

@requires_gurobi
class TestFrankWolfeBatched:

    def _ks(self):
        return knapsackModel(weights=[[3.0, 4.0, 2.0, 5.0]], capacity=[7.0])

    def test_extreme_theta_collapses_to_vertex(self):
        from pyepo.func.regularized import regularizedFrankWolfeOpt
        m = regularizedFrankWolfeOpt(self._ks(), max_iter=10, tol=1e-6)
        theta = torch.tensor([[100.0, -100.0, -100.0, -100.0]])
        mu, vertices, weights = m._frankWolfe(theta)
        assert mu.shape == (1, 4)
        assert mu[0, 0].item() > 0.99
        assert vertices.shape[0] == 1 and vertices.shape[-1] == 4
        assert weights.shape == vertices.shape[:2]
        assert weights.sum(dim=1)[0].item() == pytest.approx(1.0, abs=1e-5)

    def test_mu_equals_weighted_sum_of_vertices(self):
        from pyepo.func.regularized import regularizedFrankWolfeOpt
        m = regularizedFrankWolfeOpt(self._ks(), max_iter=12, tol=1e-6)
        theta = torch.tensor([[1.0, 1.5, 2.0, 0.5],
                              [2.0, 1.0, 1.0, 2.0]])
        mu, V, w = m._frankWolfe(theta)
        mu_check = (w.unsqueeze(-1) * V).sum(dim=1)
        assert torch.allclose(mu, mu_check, atol=1e-5)

    def test_converged_instances_are_skipped(self, monkeypatch):
        import pyepo.func.regularized as regularized
        from pyepo.func.regularized import regularizedFrankWolfeOpt

        batch_sizes = []

        def fake_solve_or_cache(cp, module):
            batch_sizes.append(cp.shape[0])
            if len(batch_sizes) == 1:
                sol = torch.zeros_like(cp)
            else:
                sol = cp.clone()
            return sol, torch.zeros(cp.shape[0], dtype=cp.dtype, device=cp.device)

        monkeypatch.setattr(regularized, "_solve_or_cache", fake_solve_or_cache)
        m = regularizedFrankWolfeOpt(self._ks(), max_iter=3, tol=1e-6)
        theta = torch.tensor([[0.0, 0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0, 0.0]])

        m._frankWolfe(theta)

        assert batch_sizes == [2, 2, 1]


# ============================================================
# regularizedFrankWolfeOpt: lambd validation, regularization, gradient correctness
# ============================================================

@requires_gurobi
class TestRegularizedFrankWolfe:

    def _ks(self):
        return knapsackModel(weights=[[3.0, 4.0, 2.0, 5.0]], capacity=[7.0])

    def test_lambd_must_be_positive(self):
        from pyepo.func.regularized import regularizedFrankWolfeOpt
        with pytest.raises(ValueError):
            regularizedFrankWolfeOpt(self._ks(), lambd=0.0)
        with pytest.raises(ValueError):
            regularizedFrankWolfeOpt(self._ks(), lambd=-1.0)

    def test_compute_regularization_includes_lambd(self):
        from pyepo.func.regularized import regularizedFrankWolfeOpt
        m = regularizedFrankWolfeOpt(self._ks(), lambd=3.0, max_iter=5)
        y = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        # Omega(y) = (lambd / 2) * ||y||^2 = 1.5 * 0.5 = 0.75
        assert torch.allclose(m.compute_regularization(y), torch.tensor([0.75]))

    def test_smaller_lambd_closer_to_IP_vertex(self):
        # small lambd: mu approaches a conv(V) vertex; large lambd: mu stays interior
        from pyepo.func.regularized import regularizedFrankWolfeOpt
        opt = self._ks()
        # weights=[3,4,2,5], cap=7, profits=[4,3,2,1] -> IP optimum picks items {0,1}
        cp = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        ip_sol = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        out_sharp = regularizedFrankWolfeOpt(opt, lambd=0.05, max_iter=30)(cp)
        out_smooth = regularizedFrankWolfeOpt(opt, lambd=10.0, max_iter=30)(cp)
        assert (out_sharp - ip_sol).norm().item() < (out_smooth - ip_sol).norm().item()


@requires_gurobi
class TestRegularizedFrankWolfeGradient:
    """Check regularizedFrankWolfeOpt autograd against finite-difference Jacobian. Primary
    correctness gate, mirrors the pattern of TestSPOPlusGradient."""

    @pytest.mark.parametrize("lambd", [0.5, 1.0, 5.0])
    def test_backward_matches_finite_difference(self, lambd):
        from pyepo.func.regularized import regularizedFrankWolfeOpt
        opt = knapsackModel(weights=[[3.0, 4.0, 2.0, 5.0]], capacity=[7.0])
        m = regularizedFrankWolfeOpt(opt, lambd=lambd, max_iter=30, tol=1e-8)
        torch.manual_seed(0)
        # cp in [1, 3]: positive, away from active-set boundary
        cp = (torch.rand(1, 4) * 2 + 1.0).requires_grad_(True)
        target = torch.randn_like(cp)
        (target * m(cp)).sum().backward()
        analytic = cp.grad.clone()
        eps = 5e-3
        fd = torch.zeros_like(cp)
        for j in range(cp.shape[1]):
            cp_p = cp.detach().clone()
            cp_p[0, j] += eps
            cp_m = cp.detach().clone()
            cp_m[0, j] -= eps
            fd[0, j] = ((target * m(cp_p)).sum() - (target * m(cp_m)).sum()) / (2 * eps)
        np.testing.assert_allclose(analytic.numpy(), fd.numpy(), atol=5e-2,
            err_msg=f"lambd={lambd}")


@requires_gurobi
class TestRegularizedFrankWolfeFenchelYoung:

    def _ks(self):
        return knapsackModel(weights=[[3.0, 4.0, 2.0, 5.0]], capacity=[7.0])

    def test_forward_matches_half_square_fenchel_young_formula(self):
        from pyepo.func.regularized import regularizedFrankWolfeFenchelYoung
        opt = self._ks()
        lambd = 1.7
        fy = regularizedFrankWolfeFenchelYoung(
            opt, lambd=lambd, max_iter=30, tol=1e-8, reduction="none"
        )
        cp = torch.tensor([[4.0, 3.0, 2.0, 1.0],
                           [1.0, 2.0, 3.0, 4.0]])
        w = torch.tensor([[1.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 1.0]])

        loss = fy(cp, w)
        r_sol = fy._frankWolfe(cp / lambd)
        omega_w = 0.5 * lambd * (w ** 2).sum(dim=-1)
        omega_r = 0.5 * lambd * (r_sol ** 2).sum(dim=-1)
        expected = omega_w - omega_r + torch.sum(cp * (r_sol - w), dim=1)

        assert torch.allclose(loss, expected, atol=1e-5)

    def test_backward_matches_fenchel_young_subgradient(self):
        from pyepo.func.regularized import regularizedFrankWolfeFenchelYoung
        opt = self._ks()
        lambd = 1.0
        fy = regularizedFrankWolfeFenchelYoung(
            opt, lambd=lambd, max_iter=30, tol=1e-8, reduction="mean"
        )
        cp = torch.tensor([[4.0, 3.0, 2.0, 1.0],
                           [1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        w = torch.tensor([[1.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 1.0]])

        loss = fy(cp, w)
        loss.backward()

        with torch.no_grad():
            r_sol = fy._frankWolfe(cp.detach() / lambd)
            expected = (r_sol - w) / cp.shape[0]

        assert torch.allclose(cp.grad, expected, atol=1e-5)


# ============================================================
# CaVE (cone-aligned cosine) losses
# ============================================================

@requires_gurobi
class TestCaVEForward:
    """Forward-pass shape / range / reduction tests."""

    @pytest.fixture
    def setup(self):
        # 2 instances, 4-dim cost, 3 binding-constraint normals each
        model = shortestPathModel(grid=(2, 3))  # num_cost = 7 edges
        d = model.num_cost
        torch.manual_seed(0)
        pred_cost = torch.randn(2, d, requires_grad=True)
        tight_ctrs = torch.randn(2, 3, d)
        return model, pred_cost, tight_ctrs

    def test_default_returns_scalar_mean(self, setup):
        model, pred_cost, ctrs = setup
        # default (max_iter=3) = truncated CaVE+ via Clarabel
        loss = coneAlignedCosine(model, processes=1, reduction="mean")(pred_cost, ctrs)
        assert loss.dim() == 0
        # cosine-distance is in [0, 2]
        assert 0.0 - 1e-6 <= loss.item() <= 2.0 + 1e-6

    def test_reduction_none(self, setup):
        model, pred_cost, ctrs = setup
        loss = coneAlignedCosine(model, processes=1, reduction="none")(pred_cost, ctrs)
        assert loss.shape == (2,)

    def test_reduction_sum(self, setup):
        model, pred_cost, ctrs = setup
        mean = coneAlignedCosine(model, processes=1, reduction="mean")(pred_cost, ctrs)
        total = coneAlignedCosine(model, processes=1, reduction="sum")(pred_cost, ctrs)
        np.testing.assert_allclose(total.item(), mean.item() * 2, atol=1e-5)

    def test_exact_preset(self, setup):
        # closer-to-exact CaVE: raise the Clarabel iteration cap
        model, pred_cost, ctrs = setup
        loss = coneAlignedCosine(
            model, max_iter=200, processes=1, reduction="mean",
        )(pred_cost, ctrs)
        assert loss.dim() == 0
        assert 0.0 - 1e-6 <= loss.item() <= 2.0 + 1e-6

    def test_gradient_flows_to_pred_cost(self, setup):
        model, pred_cost, ctrs = setup
        loss = coneAlignedCosine(model, processes=1, reduction="mean")(pred_cost, ctrs)
        loss.backward()
        assert pred_cost.grad is not None
        assert pred_cost.grad.shape == pred_cost.shape
        # projection is a fixed target — grad finite, not all-zero
        assert torch.isfinite(pred_cost.grad).all()
