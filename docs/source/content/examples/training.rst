Training
++++++++

``pyepo.func`` extends PyTorch autograd modules to support automatic differentiation through optimization. This page collects training-loop templates for each method. See :doc:`function` for the method API and selection guide.

For a runnable walkthrough, see the `03 Training and Testing <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/03%20Training%20and%20Testing.ipynb>`_ notebook.


Common Setup
============

All examples below share the same setup: a linear prediction model trained on shortest-path data.

.. code-block:: python

   import pyepo
   import torch
   from torch import nn
   from torch.utils.data import DataLoader

   # model for shortest path
   grid = (5, 5)
   optmodel = pyepo.model.shortestPathModel(grid)

   # generate data
   num_data = 1000
   num_feat = 5
   deg = 4
   noise_width = 0.5
   x, c = pyepo.data.shortestpath.genData(num_data, num_feat, grid, deg, noise_width, seed=135)

   # dataset and data loader
   dataset = pyepo.data.dataset.optDataset(optmodel, x, c)
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

   # build linear prediction model
   predmodel = nn.Linear(5, 40)
   optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-3)

   # a positive-output predictor, for the multiplicative perturbed variants below
   positive_predmodel = nn.Sequential(nn.Linear(5, 40), nn.Softplus())

Each recipe below is a self-contained training loop: pick the method you want and copy its block as-is.


Surrogate Losses
================


Smart Predict-then-Optimize+ Loss (SPO+)
----------------------------------------

.. code-block:: python

   spo = pyepo.func.SPOPlus(optmodel, processes=2)

   num_epochs = 20
   for epoch in range(num_epochs):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           loss = spo(cp, c, w, z)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Perturbation Gradient (PG)
--------------------------

.. code-block:: python

   pg = pyepo.func.PG(optmodel, sigma=0.1, two_sides=False, processes=2)

   num_epochs = 20
   for epoch in range(num_epochs):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           loss = pg(cp, c)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Perturbed Methods
=================


Differentiable Perturbed Optimizer (DPO)
----------------------------------------

``DPO`` is the additive Gaussian version. ``DPOMul`` is the multiplicative version for sign-sensitive oracles; it requires a positive-output predictor (``positive_predmodel`` in Common Setup, a linear layer followed by ``nn.Softplus()``) so that predicted costs keep their sign.

.. code-block:: python

   # additive
   ptb = pyepo.func.DPO(optmodel, n_samples=10, sigma=0.5, processes=2)
   # multiplicative: swap predmodel for positive_predmodel below
   # ptb = pyepo.func.DPOMul(optmodel, n_samples=10, sigma=0.5, processes=2)

   criterion = nn.MSELoss()

   num_epochs = 20
   for epoch in range(num_epochs):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           we = ptb(cp)
           loss = criterion(we, w)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Perturbed Fenchel-Young Loss (PFYL)
-----------------------------------

The multiplicative variant ``PFYMul`` shares the sign convention of ``DPOMul`` and requires a positive-output predictor.

.. code-block:: python

   # additive
   pfy = pyepo.func.PFY(optmodel, n_samples=10, sigma=0.5, processes=2)
   # multiplicative: swap predmodel for positive_predmodel below
   # pfy = pyepo.func.PFYMul(optmodel, n_samples=10, sigma=0.5, processes=2)

   num_epochs = 20
   for epoch in range(num_epochs):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           loss = pfy(cp, w)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Implicit Maximum Likelihood Estimator (I-MLE)
---------------------------------------------

.. code-block:: python

   imle = pyepo.func.IMLE(optmodel, n_samples=10, sigma=1.0, lambd=10, processes=2)

   criterion = nn.L1Loss()

   num_epochs = 20
   for epoch in range(num_epochs):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           we = imle(cp)
           loss = criterion(we, w)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Adaptive Implicit Maximum Likelihood Estimator (AI-MLE)
-------------------------------------------------------

.. code-block:: python

   aimle = pyepo.func.AIMLE(optmodel, n_samples=2, sigma=1.0, processes=2)

   criterion = nn.L1Loss()

   num_epochs = 20
   for epoch in range(num_epochs):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           we = aimle(cp)
           loss = criterion(we, w)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Regularized Methods
===================


L2 Regularized Frank-Wolfe (RFWO)
---------------------------------

RFWO returns a regularized solution; solution-level MSE against :math:`\mathbf{w}^*(\mathbf{c})` matches the imitation setting in the paper.

.. code-block:: python

   rfwo = pyepo.func.RFWO(optmodel, lambd=1.0, max_iter=20, tol=1e-6, processes=2)

   criterion = nn.MSELoss()

   num_epochs = 20
   for epoch in range(num_epochs):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           wr = rfwo(cp)
           loss = criterion(wr, w)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


L2 Regularized Frank-Wolfe with Fenchel-Young Loss (RFYL)
---------------------------------------------------------

.. code-block:: python

   rfyl = pyepo.func.RFY(optmodel, lambd=1.0, max_iter=20, tol=1e-6, processes=2)

   num_epochs = 20
   for epoch in range(num_epochs):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           loss = rfyl(cp, w)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Black-Box Methods
=================

Black-box methods return a predicted solution. An objective-value loss :math:`|\langle \mathbf{c}, \hat{\mathbf{w}} \rangle - z^*|` is the standard pairing.


Differentiable Black-Box Optimizer (DBB)
----------------------------------------

.. code-block:: python

   dbb = pyepo.func.DBB(optmodel, lambd=10, processes=2)

   criterion = nn.L1Loss()

   num_epochs = 20
   for epoch in range(num_epochs):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           wp = dbb(cp)
           zp = (wp * c).sum(1).view(-1, 1)
           loss = criterion(zp, z)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Negative Identity Backpropagation (NID)
---------------------------------------

NID is hyperparameter-free and uses the same training loop as DBB.

.. code-block:: python

   nid = pyepo.func.NID(optmodel, processes=2)

   criterion = nn.L1Loss()

   num_epochs = 20
   for epoch in range(num_epochs):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           wp = nid(cp)
           zp = (wp * c).sum(1).view(-1, 1)
           loss = criterion(zp, z)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Cone-Aligned Estimation
=======================


Cone-Aligned Vector Estimation (CaVE)
-------------------------------------

CaVE requires a dedicated dataset class that extracts binding-constraint normals at the optimum (``optDatasetConstrs``) and a custom collate function (``collate_tight_constraints``) to handle ragged per-instance constraint counts. The batch yields an extra ``tight_ctrs`` element on top of the usual ``(x, c, w, z)``.

.. code-block:: python

   from pyepo.data.dataset import optDatasetConstrs, collate_tight_constraints

   dataset = optDatasetConstrs(optmodel, x, c)
   dataloader = DataLoader(
       dataset, batch_size=32, shuffle=True, collate_fn=collate_tight_constraints,
   )

   cave = pyepo.func.CaVE(optmodel, processes=2)

   num_epochs = 20
   for epoch in range(num_epochs):
       for x, c, w, z, tight_ctrs in dataloader:
           cp = predmodel(x)
           loss = cave(cp, tight_ctrs)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Contrastive Methods
===================

Contrastive methods train against a cached pool of solutions. ``solve_ratio`` controls how often new instances are solved exactly during training, and ``dataset`` seeds the pool with optimal solutions. See :doc:`pool` for details on the solution-pool mechanism.


Noise Contrastive Estimation (NCE)
----------------------------------

.. code-block:: python

   nce = pyepo.func.NCE(optmodel, processes=2, solve_ratio=0.05, dataset=dataset)

   num_epochs = 20
   for epoch in range(num_epochs):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           loss = nce(cp, w)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Contrastive MAP (CMAP)
----------------------

.. code-block:: python

   cmap = pyepo.func.CMAP(optmodel, processes=2, solve_ratio=0.05, dataset=dataset)

   num_epochs = 20
   for epoch in range(num_epochs):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           loss = cmap(cp, w)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Learning to Rank (LTR)
======================

LTR variants share the same pool configuration as the contrastive methods (``solve_ratio``, ``dataset``). Pick one based on how it scores the ranking.


Pointwise LTR
-------------

.. code-block:: python

   ltr = pyepo.func.ptLTR(optmodel, processes=2, solve_ratio=0.05, dataset=dataset)

   num_epochs = 20
   for epoch in range(num_epochs):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           loss = ltr(cp, c)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Pairwise LTR
------------

.. code-block:: python

   ltr = pyepo.func.prLTR(optmodel, processes=2, solve_ratio=0.05, dataset=dataset)

   num_epochs = 20
   for epoch in range(num_epochs):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           loss = ltr(cp, c)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Listwise LTR
------------

.. code-block:: python

   ltr = pyepo.func.lsLTR(optmodel, processes=2, solve_ratio=0.05, dataset=dataset)

   num_epochs = 20
   for epoch in range(num_epochs):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           loss = ltr(cp, c)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
