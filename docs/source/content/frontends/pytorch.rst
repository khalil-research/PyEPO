PyTorch Frontend
++++++++++++++++

``pyepo.func`` provides the PyTorch training methods. Each method is an
autograd module that wraps an ``optModel``. The forward pass solves the
optimization problem; the backward pass applies the method's gradient rule.
Training uses standard PyTorch optimizers.

Method selection and per-method training loops are in
:doc:`../getting_started/function`; this page shows the calling conventions.


Training
========

End-to-end training of a shortest-path predictor on a 5x5 grid with the SPO+
loss:

.. code-block:: python

   import torch
   from torch import nn
   from torch.utils.data import DataLoader

   import pyepo
   from pyepo.data.dataset import optDataset

   # optimization model: 5x5 grid shortest path
   grid = (5, 5)
   optmodel = pyepo.model.shortestPathModel(grid)

   # synthetic data
   x, c = pyepo.data.shortestpath.genData(
       num_data=1000, num_features=5, grid=grid, deg=4, noise_width=0.5, seed=135,
   )
   dataset = optDataset(optmodel, x, c)
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

   # linear predictor and SPO+ loss
   predmodel = nn.Linear(5, optmodel.num_cost)
   spo = pyepo.func.SPOPlus(optmodel, processes=1)
   optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-3)

   # end-to-end training
   for epoch in range(10):
       for xb, cb, wb, zb in dataloader:
           loss = spo(predmodel(xb), cb, wb, zb)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

The ``optDataset`` batch is ``(x, c, w, z)``:

* ``x``: input features for the prediction model.
* ``c``: ground-truth objective coefficients.
* ``w``: optimal solution under ``c``.
* ``z``: optimal objective value under ``c``.

Some methods use only a subset of these values; the per-method inputs are
listed in the summary table of :doc:`../getting_started/function`.


Solution-Returning Modules
==========================

Solution-returning modules such as ``DPO`` are trained through a task loss on
their output. The perturbed modules draw noise internally; pass ``seed=`` for
reproducibility.

.. code-block:: python

   dpo = pyepo.func.DPO(optmodel, n_samples=10, sigma=0.5, processes=1)
   criterion = nn.MSELoss()

   for epoch in range(10):
       for xb, cb, wb, zb in dataloader:
           we = dpo(predmodel(xb))     # expected perturbed solutions
           loss = criterion(we, wb)    # task loss on the solutions
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


GPU
===

The predictor and the batch can live on CUDA; the optimization solve always
runs on the CPU. The losses expect all tensor inputs on one device, so move
the whole batch. The solver receives CPU copies internally, and the loss and
gradients come back on the batch's device:

.. code-block:: python

   device = "cuda"
   predmodel = predmodel.to(device)

   for xb, cb, wb, zb in dataloader:
       xb, cb, wb, zb = (t.to(device) for t in (xb, cb, wb, zb))
       loss = spo(predmodel(xb), cb, wb, zb)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()


Evaluation
==========

``pyepo.metric.regret`` evaluates decision quality, usually on a held-out
test set:

.. code-block:: python

   total_regret = pyepo.metric.regret(predmodel, optmodel, testloader)


Shared Options
==============

Common constructor options:

* ``processes`` controls the worker pool used for batch solving.
* ``solve_ratio`` enables solution-pool caching when set below ``1`` (CaVE repurposes this knob for its projection branch; see :doc:`../advanced/cave`).
* ``dataset`` seeds the solution pool whenever ``solve_ratio < 1``; contrastive and ranking methods require it always.
* ``reduction`` controls how per-instance losses are aggregated when the method supports it.
