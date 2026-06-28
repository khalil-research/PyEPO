Data
++++

``pyepo.data`` provides synthetic data generators and the ``optDataset`` class for wrapping data samples.

For more details, see the `02 Optimization Dataset <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/02%20Optimization%20Dataset.ipynb>`_ notebook.


Data Generator
==============

``pyepo.data`` includes synthetic data generators for four optimization problems: shortest path, multi-dimensional knapsack, traveling salesperson, and portfolio optimization.

Each generator produces feature-cost pairs :math:`(\mathbf{x}, \mathbf{c})`. The feature vector :math:`\mathbf{x}_i \in \mathbb{R}^p` follows a standard multivariate Gaussian distribution :math:`\mathcal{N}(0, \mathbf{I})`, and the cost :math:`\mathbf{c}_i \in \mathbb{R}^d` is computed from a polynomial function :math:`f(\mathbf{x}_i)` scaled by a multiplicative noise factor :math:`\boldsymbol{\epsilon}_i \sim U(1-\bar{\epsilon}, 1+\bar{\epsilon})`.

Common parameters across all generators:

* **num_data** (:math:`n`): number of data samples

* **num_features** (:math:`p`): feature dimension

* **deg** (:math:`deg`): polynomial degree of the mapping :math:`f(\mathbf{x}_i)`

* **noise_width** (:math:`\bar{\epsilon}`): noise half-width (shortest path, knapsack, TSP; portfolio uses ``noise_level`` instead; see below)

* **seed**: random seed for reproducibility


Shortest Path
-------------

A random matrix :math:`\mathcal{B} \in \mathbb{R}^{d \times p}` with Bernoulli(0.5) entries maps the feature vector into the cost coefficients: :math:`c_i^j = \big[\tfrac{1}{{3.5}^{deg}} \big(\tfrac{1}{\sqrt{p}}(\mathcal{B} \mathbf{x}_i)_j + 3\big)^{deg} + 1\big] \cdot \epsilon_i^j`.

.. autofunction:: pyepo.data.shortestpath.genData
    :noindex:

.. code-block:: python

   import pyepo

   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   grid = (5,5) # grid size
   x, c = pyepo.data.shortestpath.genData(num_data, num_feat, grid, deg=4, noise_width=0, seed=135)


Knapsack
--------

Only the cost coefficients are uncertain; item weights are fixed. Let :math:`m` be the number of items and :math:`k` the number of resource dimensions. The weights :math:`\mathcal{W} \in \mathbb{R}^{k \times m}` are sampled from 3 to 8 with one decimal place of precision. The cost coefficients are :math:`c_i^j = \big\lceil \big[\tfrac{5}{{3.5}^{deg}} \big(\tfrac{1}{\sqrt{p}}(\mathcal{B} \mathbf{x}_i)_j + 3\big)^{deg} + 1\big] \cdot \epsilon_i^j \big\rceil`.

.. autofunction:: pyepo.data.knapsack.genData
    :noindex:

.. code-block:: python

   import pyepo

   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   num_item = 32 # number of items
   dim = 3 # dimension of knapsack
   weights, x, c = pyepo.data.knapsack.genData(num_data, num_feat, num_item, dim, deg=4, noise_width=0, seed=135)


Traveling Salesperson
---------------------

The distance matrix has two components: a Euclidean distance term and a feature-encoded term. Coordinates are drawn from a mixture of a Gaussian distribution :math:`\mathcal{N}(0, \mathbf{I})` and a uniform distribution :math:`\mathbf{U}(-2, 2)`. The feature-encoded component is :math:`\tfrac{1}{{3}^{deg - 1}} \big(\tfrac{1}{\sqrt{p}} (\mathcal{B} \mathbf{x}_i)_j + 3\big)^{deg} \cdot \boldsymbol{\epsilon}_i`, where the elements of :math:`\mathcal{B}` are products of Bernoulli :math:`\mathbf{B}(0.5)` and uniform :math:`\mathbf{U}(-2, 2)` samples.

.. autofunction:: pyepo.data.tsp.genData
    :noindex:

.. code-block:: python

   import pyepo

   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   num_node = 20 # number of nodes
   x, c = pyepo.data.tsp.genData(num_data, num_feat, num_node, deg=4, noise_width=0, seed=135)


Portfolio
---------

Let :math:`\bar{r}_{ij} = \big(\tfrac{0.05}{\sqrt{p}}(\mathcal{B} \mathbf{x}_i)_j + {0.1}^{\frac{1}{deg}}\big)^{deg}`. The expected return is :math:`\mathbf{r}_i = \bar{\mathbf{r}}_i + \mathbf{L} \mathbf{f} + 0.01 \tau \boldsymbol{\epsilon}`, and the covariance matrix is :math:`\mathbf{\Sigma} = \mathbf{L} \mathbf{L}^{\intercal} + (0.01 \tau)^2 \mathbf{I}`, where :math:`\mathcal{B}` follows a Bernoulli distribution, :math:`\mathbf{L} \sim \mathbf{U}(-0.0025\tau, 0.0025\tau)`, and :math:`\mathbf{f}, \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})`.

Unlike the other generators, portfolio noise is controlled by **noise_level** (:math:`\tau`), which scales both the factor loadings :math:`\mathbf{L}` and the residual noise; ``noise_width`` does not apply here.

.. autofunction:: pyepo.data.portfolio.genData
    :noindex:

.. code-block:: python

   import pyepo

   num_data = 1000 # number of data
   num_feat = 4 # size of feature
   num_assets = 50 # number of assets
   cov, x, r = pyepo.data.portfolio.genData(num_data, num_feat, num_assets, deg=4, noise_level=1, seed=135)


Built-in Problem Models
=======================

``PyEPO`` ships ready-made models for several classic problems, so you can run the full pipeline without writing one. Each is built by a factory that takes a ``backend`` keyword (default ``"gurobi"``). Pair one with generated data and an ``optDataset``:

.. code-block:: python

   import pyepo
   from pyepo import model

   grid = (5, 5)
   x, c = pyepo.data.shortestpath.genData(1000, 5, grid, deg=4, seed=135)
   optmodel = model.shortestPathModel(grid)                  # default Gurobi
   dataset = pyepo.data.dataset.optDataset(optmodel, x, c)

Switch the solver with ``backend``. The generic backends take a ``solver=`` argument naming the open solver to run:

.. code-block:: python

   model.shortestPathModel(grid, backend="copt")
   model.shortestPathModel(grid, backend="pyomo", solver="glpk")
   model.shortestPathModel(grid, backend="mpax")             # LP on GPU

.. note:: In end-to-end training, ``pyepo.func`` modules call ``setObj`` and ``solve`` during the forward pass.


Shortest Path Model
-------------------

Minimum-cost path from the northwest to the southeast corner of an ``(h, w)`` grid, formulated as a minimum-cost-flow LP. Backends: gurobi, copt, pyomo, ortools, mpax.

.. autofunction:: pyepo.model.shortestPathModel


Knapsack Model
--------------

Multi-dimensional 0/1 knapsack: maximize value subject to per-dimension capacities. ``weights`` has shape ``(dim, n_items)`` and ``capacity`` has length ``dim``. Backends: gurobi, copt, pyomo, ortools, mpax (LP relaxation).

.. code-block:: python

   weights = [[3, 4, 3, 6, 4], [4, 5, 2, 3, 5], [5, 4, 6, 2, 3]]
   capacity = [12, 10, 15]
   optmodel = model.knapsackModel(weights, capacity)

.. autofunction:: pyepo.model.knapsackModel


Traveling Salesperson Model
---------------------------

Shortest tour visiting each city once. ``formulation`` is ``"DFJ"`` (lazy subtour elimination), ``"GG"``, or ``"MTZ"``. Backends: gurobi and copt (all three); pyomo (GG, MTZ). On gurobi, ``recycle_cuts=True`` keeps the subtour cuts found in one solve for later solves — a worthwhile speedup when the same model is re-solved many times during training.

.. code-block:: python

   optmodel = model.tspModel(20, formulation="DFJ", recycle_cuts=True)

.. autofunction:: pyepo.model.tspModel


Capacitated Vehicle Routing Model
---------------------------------

Shortest vehicle routes from a depot that serve every customer within capacity. ``formulation`` is ``"RCI"`` (lazy rounded-capacity cuts) or ``"MTZ"``. Backends: gurobi and copt (both); pyomo (MTZ). On gurobi, ``"RCI"`` also accepts ``recycle_cuts=True`` to keep the cuts found in one solve for later solves.

.. code-block:: python

   optmodel = model.vrpModel(10, demands=[2, 1, 3, 2, 1, 2, 1, 3, 2],
                             capacity=5.0, num_vehicle=3, formulation="RCI")

.. autofunction:: pyepo.model.vrpModel


Portfolio Model
---------------

Mean-variance allocation that maximizes return under a risk budget. Backends: gurobi, copt, pyomo.

.. code-block:: python

   import numpy as np
   covariance = np.cov(np.random.randn(10, 50), rowvar=False)
   optmodel = model.portfolioModel(50, covariance)

.. autofunction:: pyepo.model.portfolioModel


optDataset
==========

``pyepo.data.optDataset`` is a PyTorch ``Dataset`` that stores features and cost coefficients, and **solves the optimization problem to obtain optimal solutions and objective values**.

``optDataset`` is the standard input format for end-to-end training in ``PyEPO``: it precomputes :math:`\mathbf{w}^*(\mathbf{c})` and :math:`z^*(\mathbf{c})` once at construction time, so the training loop does not pay solver cost for these label lookups. If those labels are already available from another source, ``optDataset`` can be skipped and batches fed directly to ``pyepo.func`` modules.

.. autoclass:: pyepo.data.dataset.optDataset
    :noindex:

The following example shows how to use ``optDataset`` with a PyTorch ``DataLoader``:

.. code-block:: python

   import pyepo
   from torch.utils.data import DataLoader

   # model for shortest path
   grid = (5,5) # grid size
   model = pyepo.model.shortestPathModel(grid)

   # generate data
   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   deg = 4 # polynomial degree
   noise_width = 0 # noise width
   x, c = pyepo.data.shortestpath.genData(num_data, num_feat, grid, deg, noise_width, seed=135)

   # build dataset
   dataset = pyepo.data.dataset.optDataset(model, x, c)

   # get data loader
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


optDatasetKNN
=============

``pyepo.data.optDatasetKNN`` is a PyTorch ``Dataset`` that implements the k-nearest neighbors (kNN) robust loss [#f1]_ for decision-focused learning. It stores features and cost coefficients, and computes the **mean k-nearest-neighbor solutions and the corresponding optimal objective values**.

For a runnable walkthrough, see the `08 kNN Robust Losses <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/08%20kNN%20Robust%20Losses.ipynb>`_ notebook.

.. autoclass:: pyepo.data.dataset.optDatasetKNN
    :noindex:

.. code-block:: python

  import pyepo
  from torch.utils.data import DataLoader

  # model for shortest path
  grid = (5,5) # grid size
  model = pyepo.model.shortestPathModel(grid)

  # generate data
  num_data = 1000 # number of data
  num_feat = 5 # size of feature
  deg = 4 # polynomial degree
  noise_width = 0 # noise width
  x, c = pyepo.data.shortestpath.genData(num_data, num_feat, grid, deg, noise_width, seed=135)

  # build dataset
  dataset = pyepo.data.dataset.optDatasetKNN(model, x, c, k=10, weight=0.5)

  # get data loader
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

optDatasetConstrs
=================

``pyepo.data.dataset.optDatasetConstrs`` is a PyTorch ``Dataset`` for the CaVE [#f2]_ cone-aligned loss. In addition to the features, costs, optimal solutions, and objective values stored by ``optDataset``, it also extracts the **normals of the binding constraints at the optimal vertex** for each instance. CaVE then projects the sense-flipped predicted cost vector onto the cone spanned by these normals.

Because the binding-constraint extraction relies on Gurobi's sparse-matrix and constraint-sense APIs, ``optDatasetConstrs`` currently requires a Gurobi-backed ``optModel``. The dataset also enforces that the optimal vertex is binary, since CaVE is defined for binary linear programs.

For a runnable walkthrough that uses ``optDatasetConstrs`` end-to-end with the CaVE loss, see the `04 CaVE for Binary Linear Programs <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/04%20CaVE%20for%20Binary%20Linear%20Programs.ipynb>`_ notebook.

.. autoclass:: pyepo.data.dataset.optDatasetConstrs
    :noindex:

Per-instance constraint matrices have different row counts (different sets of constraints are tight at different vertices), so batching requires a custom ``collate_fn``:

.. autofunction:: pyepo.data.dataset.collate_tight_constraints
    :noindex:

.. code-block:: python

  import pyepo
  from torch.utils.data import DataLoader
  from pyepo.data.dataset import optDatasetConstrs, collate_tight_constraints

  # model for TSP (Gurobi backend required)
  model = pyepo.model.tspModel(num_nodes=10, formulation="DFJ")

  # generate data
  x, c = pyepo.data.tsp.genData(num_data=1000, num_features=5, num_nodes=10, deg=4, seed=135)

  # build CaVE dataset (extracts tight binding-constraint normals at the optimum)
  dataset = optDatasetConstrs(model, x, c)

  # collate_fn pads ragged per-instance constraint matrices
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_tight_constraints)

.. [#f1] Schutte, N., Postek, K., & Yorke-Smith, N. (2023). Robust Losses for Decision-Focused Learning. arXiv preprint arXiv:2310.04328.
.. [#f2] Tang, B., & Khalil, E. B. (2024). CaVE: A Cone-Aligned Approach for Fast Predict-then-Optimize with Binary Linear Programs. In Integration of Constraint Programming, Artificial Intelligence, and Operations Research (pp. 193-210).
