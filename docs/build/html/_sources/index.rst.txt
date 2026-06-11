.. PyEPO documentation master file, created by
   sphinx-quickstart on Mon Aug  9 14:15:50 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ./images/logo1.png
   :width: 1000

Welcome to PyEPO's documentation!
=================================
This is the documentation of ``PyEPO`` (PyTorch-based End-to-End Predict-then-Optimize Library), which provides decision-focused learning methods for predict-then-optimize tasks.


Quick Example
+++++++++++++

End-to-end training of a knapsack predictor defined with the DSL, using the SPO+ loss:

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

Prefer JAX? ``pyepo.func.jax`` mirrors every loss for ``jax.grad``-based end-to-end training — see :doc:`content/examples/jax`.

New to PyEPO? Start with :doc:`content/intro` for the framework, then the *Where to Start* guide in :doc:`content/tutorial`.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   content/intro
   content/install
   content/tutorial
   content/ref


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
