"""Regression contracts for MPAX storage, compilation and bounded batching."""

import numpy as np
import pytest
import torch

from .conftest import requires_mpax

pytestmark = requires_mpax


def test_dense_lp_construction_does_not_allocate_quadratic_matrix():
    import jax

    from pyepo.model.mpax.knapsack import knapsackModel

    before = {id(array) for array in jax.live_arrays()}
    model = knapsackModel(np.ones((1, 257)), [100])
    assert model.Q is None
    assert not any(
        array.shape == (257, 257) and id(array) not in before for array in jax.live_arrays()
    )


def test_grid_storage_scales_with_arcs():
    from jax.experimental.sparse import BCOO

    from pyepo.model.mpax.shortestpath import shortestPathModel

    model = shortestPathModel((20, 20))
    assert isinstance(model.A, BCOO)
    assert model.A.nse == 1520
    assert model.G.nse == 0
    np.testing.assert_allclose(np.asarray(model.A.sum(axis=0).todense()), 0)


def test_dsl_preserves_sparse_constraints_and_quadratic_objective():
    from jax.experimental.sparse import BCOO

    from pyepo import dsl

    x, c = dsl.Variable(4, lb=0, ub=3), dsl.Parameter(4)
    model = dsl.Problem(dsl.Minimize(c @ x + x @ np.eye(4) @ x), [x.sum() >= 1]).compile(
        backend="mpax"
    )
    assert isinstance(model.G, BCOO)
    assert isinstance(model.Q, BCOO)
    assert model.Q.nse == 4
    model.setObj([-2.0] * 4)
    sol, obj = model.solve()
    np.testing.assert_allclose(sol.cpu(), [1.0] * 4, atol=1e-3)
    assert obj == pytest.approx(-4.0, abs=1e-3)
    cut = model.addConstr([1.0] * 4, 2.0).addConstr([1.0, 0, 0, 0], 0.25)
    assert isinstance(cut.G, BCOO)
    cut.setObj([-2.0] * 4)
    sol, _ = cut.solve()
    assert float(sol.sum()) <= 2.001
    assert float(sol[0]) <= 0.251


@pytest.mark.parametrize("compiled", [False, True])
def test_first_torch_objective_keeps_existing_compilation(compiled):
    import jax
    import jax.numpy as jnp

    from pyepo import dsl
    from pyepo.model.mpax.shortestpath import shortestPathModel

    if compiled:
        x, c = dsl.Variable(2, lb=0, ub=1), dsl.Parameter(2)
        model = dsl.Problem(dsl.Minimize(c @ x), [x.sum() >= 1]).compile("mpax")
    else:
        model = shortestPathModel((2, 2))
    costs = jnp.ones((2, model.num_cost), dtype=jnp.float32)
    jax.block_until_ready(model.batch_optimize(costs))
    solve = model.batch_optimize
    model.setObj(torch.ones((2, model.num_cost)))
    assert model.batch_optimize is solve
    assert solve._cache_size() == 1


@pytest.mark.parametrize("tensor_costs", [False, True])
def test_dataset_chunks_preserve_order_and_objective_offset(tensor_costs):
    from pyepo import dsl
    from pyepo.data.dataset import optDataset

    x, c = dsl.Variable(2, lb=0, ub=1), dsl.Parameter(2)
    model = dsl.Problem(dsl.Maximize(c @ x + ((0 * x).sum() + 7)), [x.sum() <= 1]).compile("mpax")
    costs = np.array([[4, 1], [1, 5], [6, 1], [1, 7], [8, 1]], dtype=np.float32)
    if tensor_costs:
        costs = torch.from_numpy(costs)
    dataset = optDataset(model, np.zeros((5, 1)), costs, solve_batch_size=2)
    np.testing.assert_allclose(dataset.sols, [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]], atol=1e-3)
    np.testing.assert_allclose(dataset.objs[:, 0], [11, 12, 13, 14, 15], atol=1e-3)
    assert model.c.shape[0] == 2
    assert model.batch_optimize._cache_size() == 1


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_dataset_rejects_invalid_solve_batch_size(size):
    from pyepo.data.dataset import optDataset
    from pyepo.model.mpax.shortestpath import shortestPathModel

    with pytest.raises(ValueError, match="solve_batch_size"):
        optDataset(
            shortestPathModel((2, 2)), np.zeros((1, 1)), np.ones((1, 4)), solve_batch_size=size
        )
