Training
++++++++

``pyepo.func`` extends PyTorch autograd modules to support automatic differentiation through optimization. This enables end-to-end training of neural networks for predict-then-optimize problems.

For more details, see the `03 Training and Testing <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/03%20Training%20and%20Testing.ipynb>`_ notebook.


Common Setup
============

All training examples below share the same setup: a linear prediction model trained on shortest path data. The setup code is shown once here and omitted in subsequent sections.

.. code-block:: python

   import pyepo
   import torch
   from torch import nn
   from torch.utils.data import DataLoader

   # model for shortest path
   grid = (5,5) # grid size
   optmodel = pyepo.model.grb.shortestPathModel(grid)

   # generate data
   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   deg = 4 # polynomial degree
   noise_width = 0.5 # noise width
   x, c = pyepo.data.shortestpath.genData(num_data, num_feat, grid, deg, noise_width, seed=135)

   # build dataset
   dataset = pyepo.data.dataset.optDataset(optmodel, x, c)

   # get data loader
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

   # build linear prediction model
   class LinearRegression(nn.Module):

       def __init__(self):
           super().__init__()
           self.linear = nn.Linear(5, 40)

       def forward(self, x):
           out = self.linear(x)
           return out

   # init
   predmodel = LinearRegression()
   # set optimizer
   optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-3)


Training with SPO+
==================

SPO+ is a surrogate loss that directly measures decision quality. It takes predicted costs, true costs, optimal solutions, and optimal objective values.

.. code-block:: python

   # init SPO+ loss
   spo = pyepo.func.SPOPlus(optmodel, processes=2)

   # training
   num_epochs = 20
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           # forward pass
           cp = predmodel(x)
           # SPO+ loss
           loss = spo(cp, c, w, z)
           # backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Training with DBB
=================

The differentiable black-box optimizer replaces zero gradients with interpolated gradients. It returns predicted solutions, which are then used to compute an objective-value-based loss.

.. code-block:: python

   # init black-box optimizer
   dbb = pyepo.func.blackboxOpt(optmodel, lambd=10, processes=2)
   # init loss
   criterion = nn.L1Loss()

   # training
   num_epochs = 20
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           # forward pass
           cp = predmodel(x)
           # black-box optimizer
           wp = dbb(cp)
           # objective value
           zp = (wp * c).sum(1).view(-1, 1)
           # regret loss
           loss = criterion(zp, z)
           # backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Training with NID
=================

Negative Identity Backpropagation treats the solver as a negative identity during backpropagation. The training loop follows the same pattern as DBB.

.. code-block:: python

   # init NID optimizer
   nid = pyepo.func.negativeIdentity(optmodel, processes=2)
   # init loss
   criterion = nn.L1Loss()

   # training
   num_epochs = 20
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           # forward pass
           cp = predmodel(x)
           # NID optimizer
           wp = nid(cp)
           # objective value
           zp = (wp * c).sum(1).view(-1, 1)
           # regret loss
           loss = criterion(zp, z)
           # backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Training with DPO
=================

The differentiable perturbed optimizer (DPO) returns the expected solution under a perturbed distribution, so the training loop must choose a task loss, such as MSE between the expected solution and the true optimal solution. ``perturbedOpt`` is the additive Gaussian version. ``perturbedOptMul`` uses multiplicative perturbations and is useful when the oracle is sign-sensitive and cost entries must keep their signs. The multiplicative variant assumes predicted costs already have the intended nonzero sign; for nonnegative-cost problems, use a positive-output predictor such as Softplus plus a small epsilon.

Additive perturbation:

.. code-block:: python

   # init additive DPO optimizer
   ptb = pyepo.func.perturbedOpt(optmodel, n_samples=10, sigma=0.5, processes=2)
   # init loss
   criterion = nn.MSELoss()

   # training
   num_epochs = 20
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           # forward pass
           cp = predmodel(x)
           # perturbed optimizer
           we = ptb(cp)
           # MSE loss
           loss = criterion(we, w)
           # backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

Multiplicative perturbation:

Here ``positive_predmodel`` denotes a predictor whose output is constrained to the valid cost sign, for example by applying ``nn.Softplus()`` and adding a small epsilon.

.. code-block:: python

   # init multiplicative DPO optimizer
   ptb = pyepo.func.perturbedOptMul(optmodel, n_samples=10, sigma=0.5, processes=2)
   # init loss
   criterion = nn.MSELoss()

   # training
   num_epochs = 20
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           # forward pass
           cp = positive_predmodel(x)
           # perturbed optimizer
           we = ptb(cp)
           # MSE loss
           loss = criterion(we, w)
           # backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Training with PFYL
==================

Perturbed Fenchel-Young loss (PFYL) uses the same perturbed expected solution internally, but returns the Fenchel-Young loss directly from predicted costs and true optimal solutions. It does not require a separate task loss. ``perturbedFenchelYoung`` is the additive Gaussian version. ``perturbedFenchelYoungMul`` is the multiplicative sign-preserving variant of PFYL. Like ``perturbedOptMul``, it assumes predicted costs already have the intended nonzero sign.

Additive perturbation:

.. code-block:: python

   # init additive PFYL loss
   pfy = pyepo.func.perturbedFenchelYoung(optmodel, n_samples=10, sigma=0.5, processes=2)

   # training
   num_epochs = 20
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           # forward pass
           cp = predmodel(x)
           # Fenchel-Young loss
           loss = pfy(cp, w)
           # backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

Multiplicative perturbation:

Here ``positive_predmodel`` denotes a predictor whose output is constrained to the valid cost sign, for example by applying ``nn.Softplus()`` and adding a small epsilon.

.. code-block:: python

   # init multiplicative PFYL loss
   pfy = pyepo.func.perturbedFenchelYoungMul(optmodel, n_samples=10, sigma=0.5, processes=2)

   # training
   num_epochs = 20
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           # forward pass
           cp = positive_predmodel(x)
           # Fenchel-Young loss
           loss = pfy(cp, w)
           # backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Training with RFWO
==================

The regularized Frank-Wolfe optimizer returns a smooth solution in the convex hull of feasible solutions. Since it returns a solution rather than a loss, define a task loss explicitly, for example MSE between the regularized solution and the true optimal solution.

.. code-block:: python

   rfwo = pyepo.func.regularizedFrankWolfeOpt(
       optmodel, lambd=1.0, max_iter=20, tol=1e-6, processes=2)
   criterion = nn.MSELoss()

   def rfwomse(cp, w):
       wr = rfwo(cp)
       return criterion(wr, w)


Training with RFYL
==================

Regularized Frank-Wolfe Fenchel-Young loss computes the L2 regularized Fenchel-Young objective directly from predicted costs and true optimal solutions.

.. code-block:: python

   rfyl = pyepo.func.regularizedFrankWolfeFenchelYoung(
       optmodel, lambd=1.0, max_iter=20, tol=1e-6, processes=2)

   loss = rfyl(cp, w)


Training with NCE
=================

Noise Contrastive Estimation uses a set of non-optimal solutions as negative samples. It takes predicted costs and true costs as input.

.. code-block:: python

   # init NCE loss
   nce = pyepo.func.NCE(optmodel, processes=2, solve_ratio=0.05, dataset=dataset)

   # training
   num_epochs = 20
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           # forward pass
           cp = predmodel(x)
           # NCE loss
           loss = nce(cp, c)
           # backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Training with LTR
=================

Learning-to-Rank methods learn a scoring function that ranks feasible solutions. Three variants are available: pointwise, pairwise, and listwise.

.. code-block:: python

   # init LTR loss (choose one)
   # pointwise
   #ltr = pyepo.func.pointwiseLTR(optmodel, processes=2, solve_ratio=0.05, dataset=dataset)
   # pairwise
   #ltr = pyepo.func.pairwiseLTR(optmodel, processes=2, solve_ratio=0.05, dataset=dataset)
   # listwise
   ltr = pyepo.func.listwiseLTR(optmodel, processes=2, solve_ratio=0.05, dataset=dataset)

   # training
   num_epochs = 20
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           # forward pass
           cp = predmodel(x)
           # LTR loss
           loss = ltr(cp, c)
           # backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Training with I-MLE
===================

I-MLE uses the perturb-and-MAP framework with Sum-of-Gamma noise. It returns perturbed solutions and uses loss interpolation to approximate gradients.

.. code-block:: python

   # init I-MLE optimizer
   imle = pyepo.func.implicitMLE(optmodel, n_samples=10, sigma=1.0, lambd=10, two_sides=False, processes=2)
   # init loss
   criterion = nn.L1Loss()

   # training
   num_epochs = 20
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           # forward pass
           cp = predmodel(x)
           # I-MLE optimizer
           we = imle(cp)
           # L1 loss
           loss = criterion(we, w)
           # backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Training with AI-MLE
====================

AI-MLE extends I-MLE with an adaptive interpolation step for better gradient estimates.

.. code-block:: python

   # init AI-MLE optimizer
   aimle = pyepo.func.adaptiveImplicitMLE(optmodel, n_samples=2, sigma=1.0, two_sides=True, processes=2)
   # init loss
   criterion = nn.L1Loss()

   # training
   num_epochs = 20
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           # forward pass
           cp = predmodel(x)
           # AI-MLE optimizer
           we = aimle(cp)
           # L1 loss
           loss = criterion(we, w)
           # backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Training with PG
=================

Perturbation Gradient uses zeroth-order gradient approximation via finite differences along the true cost direction. It takes predicted costs and true costs as input.

.. code-block:: python

   # init PG loss
   pg = pyepo.func.perturbationGradient(optmodel, sigma=0.1, two_sides=False, processes=2)

   # training
   num_epochs = 20
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           # forward pass
           cp = predmodel(x)
           # PG loss
           loss = pg(cp, c)
           # backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
