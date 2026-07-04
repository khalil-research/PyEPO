JAX Frontend
++++++++++++

``pyepo.func.jax`` provides JAX versions of the PyEPO training methods for use
with ``jax.grad``. Class names, constructor style, call signatures, and short
aliases follow the PyTorch frontend:

.. code-block:: python

   # torch:  from pyepo.func import SPOPlus, DPO, PFY
   # jax:    from pyepo.func.jax import SPOPlus, DPO, PFY

The losses use ``jax.custom_vjp``. The forward pass solves the optimization
model, and the backward pass applies the gradient rule for the selected method.
See :doc:`../getting_started/function` for the loss families and method inputs.


Installation
============

* ``pip install pyepo[mpax]``: the JAX loss frontend and the MPAX fast path.
* ``pip install pyepo[jaxdev]``: the Flax and optax dependencies for the
  examples below.


Solver Backends
===============

The frontend works with PyEPO solver backends:

* **MPAX** is solved natively. The PDHG solve is JAX-traceable, so the training
  step can be used with ``jax.jit``.
* **Non-MPAX backends** (GurobiPy, COPT, Pyomo, OR-Tools) are reached through
  ``jax.pure_callback``, which wraps the existing CPU solver. This path requires
  JAX plus the selected backend's solver package.

The training step can be wrapped in ``@jax.jit`` on either path. MPAX is where
``jit`` also accelerates the solve itself.


Training
========

End-to-end training of a shortest-path predictor on a 5x5 grid with the SPO+
loss, using a Flax linear layer and an Optax optimizer:

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


Jitted Training on MPAX
=======================

With the MPAX backend, the whole training step (prediction, batch solve, and
optimizer update) compiles into one ``jax.jit`` function. Continuing the setup
above, switch to an MPAX model:

.. code-block:: python

   optmodel = pyepo.model.shortestPathModel(grid, backend="mpax")
   spo = SPOPlus(optmodel, reduction="mean")

   @jax.jit
   def train_step(params, opt_state, xb, cb, wb, zb):
       loss, grads = jax.value_and_grad(
           lambda p: spo(predmodel.apply(p, xb), cb, wb, zb)
       )(params)
       updates, opt_state = optimizer.update(grads, opt_state)
       return optax.apply_updates(params, updates), opt_state, loss

   for epoch in range(10):
       params, opt_state, loss = train_step(params, opt_state, xj, cj, wj, zj)

The same ``train_step`` also compiles on a non-MPAX backend, but the solver
then runs on the CPU inside ``jax.pure_callback``.


Solution-Returning Modules and RNG Keys
=======================================

Solution-returning modules such as ``DPO`` are trained through a task loss
written in plain ``jax.numpy``. The randomized losses (the perturbed family)
draw noise internally when run eagerly. Under ``jax.jit`` they require an
explicit ``key=``, which becomes a traced argument:

.. code-block:: python

   from pyepo.func.jax import DPO

   dpo = DPO(optmodel, n_samples=10, sigma=0.5)

   def loss_fn(p, k):
       we = dpo(predmodel.apply(p, xj), key=k)   # expected perturbed solutions
       return jnp.mean((we - wj) ** 2)           # task loss on the solutions

   step = jax.jit(jax.grad(loss_fn))

   key = jax.random.PRNGKey(0)
   for epoch in range(10):
       key, subkey = jax.random.split(key)
       grads = step(params, subkey)
       updates, opt_state = optimizer.update(grads, opt_state)
       params = optax.apply_updates(params, updates)

Calling a randomized loss inside ``jax.jit`` without a ``key`` raises an
error instead of silently freezing one noise draw.


Evaluation
==========

Evaluation works as in PyTorch. ``pyepo.metric.regret`` accepts a JAX callable.
Evaluate on a held-out test set in real experiments:

.. code-block:: python

   from torch.utils.data import DataLoader

   dataloader = DataLoader(ds, batch_size=32)
   pred_fn = lambda feats: predmodel.apply(params, jnp.asarray(feats))
   total_regret = pyepo.metric.regret(pred_fn, optmodel, dataloader)


Notes
=====

* **Caching and pool growth**: solution-pool caching (``solve_ratio < 1``) and
  the online pool growth of the contrastive / ranking losses are supported, but
  they are eager-only and cannot be ``jax.jit``-ed.
* **CaVE**: the hybrid branch (``0 < solve_ratio < 1``) draws a per-batch coin
  and raises under ``jax.jit``. Run it eagerly or use ``solve_ratio`` of 0 or 1.
* **AIMLE** is eager-only. The other randomized losses are jittable with an
  explicit ``key``.
* **API**: JAX losses follow the PyTorch signatures, except ``IMLE`` /
  ``AIMLE``, which take ``kappa`` / ``n_iterations`` / ``seed`` scalars
  instead of a PyTorch ``distribution`` object.
