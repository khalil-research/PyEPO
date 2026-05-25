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
   optmodel = pyepo.model.grb.shortestPathModel(grid)

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
   class LinearRegression(nn.Module):
       def __init__(self):
           super().__init__()
           self.linear = nn.Linear(5, 40)
       def forward(self, x):
           return self.linear(x)

   predmodel = LinearRegression()
   optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-3)


Two Training Patterns
=====================

PyEPO methods fall into two interaction patterns. Both share the same outer loop; they differ in how the inner block computes the loss.


Pattern A — Loss-returning methods
----------------------------------

Methods that return a scalar loss directly (SPO+, PFYL, RFYL, NCE, CMAP, LTR, PG, CaVE) plug straight into ``.backward()``:

.. code-block:: python

   num_epochs = 20
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           cp = predmodel(x)
           loss = method(cp, ...)  # method-specific call: see recipes below
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Pattern B — Solution-returning methods
--------------------------------------

Methods that return predicted, expected, or regularized solutions (DPO, DBB, NID, RFWO, I-MLE, AI-MLE) require the user to define a task loss on the output:

.. code-block:: python

   criterion = nn.MSELoss()  # or nn.L1Loss(), or an objective-value loss

   num_epochs = 20
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           cp = predmodel(x)
           wp = method(cp)
           # solution-level loss (vs. true optimal solution):
           loss = criterion(wp, w)
           # or objective-value loss (vs. true optimal value):
           # zp = (wp * c).sum(1).view(-1, 1)
           # loss = criterion(zp, z)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


The recipes below give each method's initialization and call signature. Drop them into the corresponding pattern.


Surrogate Losses
================

Both surrogate losses follow **Pattern A**.


SPO+
----

.. code-block:: python

   spo = pyepo.func.SPOPlus(optmodel, processes=2)

   # inner call: SPO+ takes (cp, c, w, z)
   loss = spo(cp, c, w, z)


PG
--

.. code-block:: python

   pg = pyepo.func.perturbationGradient(optmodel, sigma=0.1, two_sides=False, processes=2)

   # inner call: PG takes (cp, c)
   loss = pg(cp, c)


Perturbed Methods
=================


DPO — additive and multiplicative
---------------------------------

**Pattern B.** ``perturbedOptMul`` requires a positive-output predictor (e.g., ``nn.Softplus()`` plus a small epsilon, denoted ``positive_predmodel`` below) so that predicted costs keep their sign.

.. code-block:: python

   # additive
   ptb = pyepo.func.perturbedOpt(optmodel, n_samples=10, sigma=0.5, processes=2)
   # multiplicative — use positive_predmodel for cp
   # ptb = pyepo.func.perturbedOptMul(optmodel, n_samples=10, sigma=0.5, processes=2)

   criterion = nn.MSELoss()

   # inner block
   we = ptb(cp)
   loss = criterion(we, w)


PFYL — additive and multiplicative
----------------------------------

**Pattern A.** The multiplicative variant shares the sign convention of ``perturbedOptMul``.

.. code-block:: python

   # additive
   pfy = pyepo.func.perturbedFenchelYoung(optmodel, n_samples=10, sigma=0.5, processes=2)
   # multiplicative — use positive_predmodel for cp
   # pfy = pyepo.func.perturbedFenchelYoungMul(optmodel, n_samples=10, sigma=0.5, processes=2)

   # inner call: PFYL takes (cp, w)
   loss = pfy(cp, w)


I-MLE
-----

**Pattern B.**

.. code-block:: python

   imle = pyepo.func.implicitMLE(optmodel, n_samples=10, sigma=1.0, lambd=10, processes=2)

   criterion = nn.L1Loss()

   # inner block
   we = imle(cp)
   loss = criterion(we, w)


AI-MLE
------

**Pattern B.**

.. code-block:: python

   aimle = pyepo.func.adaptiveImplicitMLE(optmodel, n_samples=2, sigma=1.0, processes=2)

   criterion = nn.L1Loss()

   # inner block
   we = aimle(cp)
   loss = criterion(we, w)


Regularized Methods
===================


RFWO
----

**Pattern B.** Solution-level MSE matches the imitation setting in the paper.

.. code-block:: python

   rfwo = pyepo.func.regularizedFrankWolfeOpt(optmodel, lambd=1.0, max_iter=20, tol=1e-6, processes=2)

   criterion = nn.MSELoss()

   # inner block
   wr = rfwo(cp)
   loss = criterion(wr, w)


RFYL
----

**Pattern A.**

.. code-block:: python

   rfyl = pyepo.func.regularizedFrankWolfeFenchelYoung(optmodel, lambd=1.0, max_iter=20, tol=1e-6, processes=2)

   # inner call: RFYL takes (cp, w)
   loss = rfyl(cp, w)


Black-Box Methods
=================

Both follow **Pattern B** with an objective-value loss (:math:`|\langle \mathbf{c}, \hat{\mathbf{w}} \rangle - z^*|`).


DBB
---

.. code-block:: python

   dbb = pyepo.func.blackboxOpt(optmodel, lambd=10, processes=2)

   criterion = nn.L1Loss()

   # inner block
   wp = dbb(cp)
   zp = (wp * c).sum(1).view(-1, 1)
   loss = criterion(zp, z)


NID
---

Hyperparameter-free; same inner block as DBB.

.. code-block:: python

   nid = pyepo.func.negativeIdentity(optmodel, processes=2)

   criterion = nn.L1Loss()

   # inner block
   wp = nid(cp)
   zp = (wp * c).sum(1).view(-1, 1)
   loss = criterion(zp, z)


Cone-Aligned Estimation
=======================


CaVE
----

CaVE requires a dedicated dataset class that extracts binding-constraint normals at the optimum (``optDatasetConstrs``) and a custom collate function (``collate_tight_constraints``) to handle ragged per-instance constraint counts. The batch yields an extra ``tight_ctrs`` element on top of the usual ``(x, c, w, z)``.

.. code-block:: python

   from pyepo.data.dataset import optDatasetConstrs, collate_tight_constraints

   dataset = optDatasetConstrs(optmodel, x_train, c_train)
   dataloader = DataLoader(
       dataset, batch_size=32, shuffle=True, collate_fn=collate_tight_constraints,
   )

   cave = pyepo.func.coneAlignedCosine(optmodel, processes=2)

   num_epochs = 20
   for epoch in range(num_epochs):
       for x, c, w, z, tight_ctrs in dataloader:
           cp = predmodel(x)
           # inner call: CaVE takes (cp, tight_ctrs)
           loss = cave(cp, tight_ctrs)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Contrastive Methods
===================

NCE and CMAP both follow **Pattern A** and use a cached solution pool: ``solve_ratio`` controls how often new instances are solved exactly during training, and ``dataset`` seeds the pool with optimal solutions.


NCE
---

.. code-block:: python

   nce = pyepo.func.NCE(optmodel, processes=2, solve_ratio=0.05, dataset=dataset)

   # inner call: NCE takes (cp, w)
   loss = nce(cp, w)


CMAP
----

.. code-block:: python

   cmap = pyepo.func.contrastiveMAP(optmodel, processes=2, solve_ratio=0.05, dataset=dataset)

   # inner call: CMAP takes (cp, w)
   loss = cmap(cp, w)


Learning to Rank
================

All three LTR variants follow **Pattern A** and share the same pool configuration. Pick one based on how it scores the ranking.


Pointwise / Pairwise / Listwise
-------------------------------

.. code-block:: python

   # pick one
   ltr = pyepo.func.pointwiseLTR(optmodel, processes=2, solve_ratio=0.05, dataset=dataset)
   # ltr = pyepo.func.pairwiseLTR(optmodel, processes=2, solve_ratio=0.05, dataset=dataset)
   # ltr = pyepo.func.listwiseLTR(optmodel, processes=2, solve_ratio=0.05, dataset=dataset)

   # inner call: LTR takes (cp, c)
   loss = ltr(cp, c)
