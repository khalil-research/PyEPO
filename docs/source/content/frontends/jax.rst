JAX Frontend
++++++++++++

``pyepo.func.jax`` provides JAX versions of the PyEPO training methods for use
with ``jax.grad``. Class names, constructor style, call signatures, and short
aliases follow the PyTorch frontend:

.. code-block:: python

   # torch:  from pyepo.func import SPOPlus
   # jax:    from pyepo.func.jax import SPOPlus

The losses use ``jax.custom_vjp``. The forward pass solves the optimization
model, and the backward pass applies the gradient rule for the selected method.
See :doc:`../getting_started/function` for the loss families and method inputs.


Solver Backends
===============

The frontend works with PyEPO solver backends:

* **MPAX** is solved natively. The PDHG solve is JAX-traceable, so the training
  step can be used with ``jax.jit``.
* **Non-MPAX backends** (GurobiPy, COPT, Pyomo, OR-Tools) are reached through
  ``jax.pure_callback``, which wraps the existing CPU solver. This path needs
  JAX plus the selected backend's solver package.


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

   # optimization model: 5x5 grid shortest path
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

Wrap the training step in ``@jax.jit`` and close over ``optmodel`` when using
the MPAX backend.


Installation
============

* ``pip install pyepo[mpax]``: the loss frontend and the MPAX fast path.
* The callback path for non-JAX backends needs JAX plus the selected backend's solver package.
* ``pip install pyepo[jaxdev]``: the Flax and optax dependencies for the
  example above.


Notes
=====

* **jax.jit**: jit the training step by closing over the model. The randomized
  losses (the perturbed family and ``implicitMLE``) are jittable when you pass an explicit ``key``;
  ``adaptiveImplicitMLE`` is eager-only.
* **Caching and pool growth**: solution-pool caching (``solve_ratio < 1``) and
  the online pool growth of the contrastive / ranking losses are supported and
  follow the PyTorch implementation, but eager-only; they cannot be ``jax.jit``-ed.
* **API**: JAX losses follow the PyTorch signatures, except ``implicitMLE`` /
  ``adaptiveImplicitMLE``, which take ``kappa`` / ``n_iterations`` / ``seed``
  scalars instead of a PyTorch ``distribution`` object.
