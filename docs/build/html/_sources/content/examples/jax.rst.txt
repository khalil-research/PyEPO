JAX Frontend
++++++++++++

``pyepo.func.jax`` mirrors ``pyepo.func`` so a JAX or Flax model can be trained
end-to-end with ``jax.grad``. Every loss keeps the same class name, constructor
and call signature, and acronym alias as its PyTorch counterpart, so switching
frameworks is a one-line import change:

.. code-block:: python

   # torch:  from pyepo.func import SPOPlus
   # jax:    from pyepo.func.jax import SPOPlus

The losses are backed by ``jax.custom_vjp``: the forward pass solves the
optimization model (a black box) and the backward pass applies the loss's
hand-derived gradient rule, so the solver never has to be differentiable. See
:doc:`function` for the loss families and how to choose one — the JAX classes
behave identically.


Solver Backends
===============

The frontend works with **any** PyEPO solver backend:

* **MPAX** is solved natively — the PDHG solve is JAX-traceable, so the whole
  training step is ``jax.jit``-able and GPU-native, with no CPU round-trip.
* **Every other backend** (GurobiPy, COPT, Pyomo, OR-Tools) is reached through
  ``jax.pure_callback``, which wraps the existing CPU solver. This is the
  universal default and needs only a JAX install.


Example
=======

End-to-end training of a shortest-path predictor on a 5x5 grid with the SPO+
loss, using a Flax linear layer and an optax optimizer:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import optax
   from flax import linen as nn

   import pyepo
   from pyepo.data.dataset import optDataset
   from pyepo.func.jax import SPOPlus

   # optimization model: 5x5 grid shortest path (any PyEPO solver works)
   grid = (5, 5)
   optmodel = pyepo.model.shortestPathModel(grid)

   # synthetic data
   x, c = pyepo.data.shortestpath.genData(
       num_data=1000, num_features=5, grid=grid, deg=4, noise_width=0.5, seed=135,
   )
   ds = optDataset(optmodel, x, c)
   xj = jnp.asarray(x, jnp.float32)
   cj, wj, zj = (jnp.asarray(a, jnp.float32) for a in (ds.costs, ds.sols, ds.objs))

   # linear predictor and SPO+ loss
   predmodel = nn.Dense(optmodel.num_cost)
   params = predmodel.init(jax.random.PRNGKey(0), xj[:1])
   spo = SPOPlus(optmodel, reduction="mean")
   optimizer = optax.adam(1e-2)
   opt_state = optimizer.init(params)

   # end-to-end training
   for epoch in range(10):
       grads = jax.grad(lambda p: spo(predmodel.apply(p, xj), cj, wj, zj))(params)
       updates, opt_state = optimizer.update(grads, opt_state)
       params = optax.apply_updates(params, updates)

Wrap the training step in ``@jax.jit`` (closing over ``optmodel``) to compile
it; on the MPAX backend it runs GPU-native.


Installation
============

* ``pip install pyepo[mpax]`` — the loss frontend and the MPAX fast path.
* The any-solver callback path needs only a JAX install.
* ``pip install pyepo[jaxdev]`` — the Flax and optax dependencies for the
  example above.


Notes
=====

* **jax.jit**: jit the whole training step by closing over the model — no
  pytree registration is needed. The randomized losses (the perturbed family
  and ``implicitMLE``) are jittable when you pass an explicit ``key``;
  ``adaptiveImplicitMLE`` is eager-only.
* **Caching and pool growth**: solution-pool caching (``solve_ratio < 1``) and
  the online pool growth of the contrastive / ranking losses are supported and
  faithful to PyTorch, but eager-only — they cannot be ``jax.jit``-ed.
* **API**: every loss keeps PyTorch's signature, except ``implicitMLE`` /
  ``adaptiveImplicitMLE``, which take ``kappa`` / ``n_iterations`` / ``seed``
  scalars instead of a PyTorch ``distribution`` object.
