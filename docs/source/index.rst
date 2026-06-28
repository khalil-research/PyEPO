.. image:: ./images/logo1.png
   :width: 1000

PyEPO Documentation
===================
``PyEPO`` is a PyTorch/JAX-based library for end-to-end predict-then-optimize training.

New to PyEPO? Start with :doc:`content/intro`, install ``PyEPO`` and a solver backend, then follow the :doc:`content/getting_started/workflow`.


Quick Example
+++++++++++++

End-to-end training of a knapsack predictor defined with the DSL and trained with the SPO+ loss:

This example uses Gurobi as the backend. If you do not have a Gurobi license, install a different PyEPO backend and change ``backend=`` accordingly.

.. code-block:: python

   import numpy as np
   import pyepo
   from pyepo import EPO, dsl
   import torch
   from torch import nn
   from torch.utils.data import DataLoader

   # generate knapsack data
   num_item = 10
   weights, feat, costs = pyepo.data.knapsack.genData(
       1000, 5, num_item, 3, deg=4, noise_width=0.5, seed=135,
   )
   capacity = (weights.sum(axis=1) * 0.5).astype(int)

   # define the problem with the DSL
   x = dsl.Variable(num_item, vtype=EPO.BINARY)
   c = dsl.Parameter(num_item)
   optmodel = dsl.Problem(dsl.Maximize(c @ x), [weights @ x <= capacity]).compile(backend="gurobi")

   # dataset
   dataset = pyepo.data.dataset.optDataset(optmodel, feat, costs)
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

   # linear predictor and SPO+ loss
   predmodel = nn.Linear(5, num_item)
   spo = pyepo.func.SPOPlus(optmodel, processes=1)
   optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-3)

   # end-to-end training
   for epoch in range(10):
       for xb, cb, wb, zb in dataloader:
           loss = spo(predmodel(xb), cb, wb, zb)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

   # decision quality (on the training set here; split off a test set for real evaluation)
   print("Training regret:", pyepo.metric.regret(predmodel, optmodel, dataloader))

Prefer JAX? ``pyepo.func.jax`` follows the PyTorch loss API for ``jax.grad``-based training; see :doc:`content/frontends/jax`.


.. toctree::
   :maxdepth: 2
   :caption: Basics

   content/intro
   content/install

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   content/getting_started/workflow
   content/getting_started/model
   content/getting_started/data
   content/getting_started/function
   content/getting_started/evaluation

.. toctree::
   :maxdepth: 2
   :caption: Frontends and Backends

   content/frontends/pytorch
   content/frontends/jax
   content/solver_backends

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   content/advanced/twostage
   content/advanced/pool
   content/advanced/cave
   content/advanced/knn

.. toctree::
   :maxdepth: 2
   :caption: Notebooks

   content/notebooks

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   content/api

.. toctree::
   :maxdepth: 2
   :caption: Citation

   content/ref


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
