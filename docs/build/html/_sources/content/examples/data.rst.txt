Data
++++

``pyepo.data`` provides synthetic data generators and the ``optDataset`` class for wrapping data samples.

For more details, see the `02 Optimization Dataset <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/02%20Optimization%20Dataset.ipynb>`_ notebook.


Data Generator
==============

``pyepo.data`` includes synthetic data generators for four optimization problems: shortest path, multi-dimensional knapsack, traveling salesperson, and portfolio optimization.

Each generator produces feature-cost pairs :math:`(\mathbf{x}, \mathbf{c})`. The feature vector :math:`\mathbf{x}_i \in \mathbb{R}^p` follows a standard multivariate Gaussian distribution :math:`\mathcal{N}(0, \mathbf{I})`, and the cost :math:`\mathbf{c}_i \in \mathbb{R}^d` is computed from a polynomial function :math:`f(\mathbf{x}_i)` multiplied by random noise :math:`\mathbf{\epsilon}_i \sim U(1-\bar{\epsilon}, 1+\bar{\epsilon})`.

Common parameters across all generators:

* **num_data** (:math:`n`): number of data samples

* **num_features** (:math:`p`): feature dimension

* **deg** (:math:`deg`): polynomial degree of the mapping :math:`f(\mathbf{x}_i)`

* **noise_width** (:math:`\bar{\epsilon}`): noise half-width

* **seed**: random seed for reproducibility


Shortest Path
-------------

A random matrix :math:`\mathcal{B} \in \mathbb{R}^{d \times p}` with Bernoulli(0.5) entries encodes the features. The cost coefficients are generated as :math:`c_i^j = [\frac{1}{{3.5}^{deg}} (\frac{1}{\sqrt{p}}(\mathcal{B} \mathbf{x}_i)_j + 3)^{deg} + 1] \cdot \epsilon_i^j`.

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

Since uncertain coefficients appear only in the objective function, item weights are fixed. Let :math:`m` be the number of items and :math:`k` the number of resource dimensions. The weights :math:`\mathcal{W} \in \mathbb{R}^{k \times m}` are sampled from 3 to 8 with one decimal place of precision. The cost coefficients are :math:`c_i^j = \lceil [\frac{5}{{3.5}^{deg}} (\frac{1}{\sqrt{p}}(\mathcal{B} \mathbf{x}_i)_j + 3)^{deg} + 1] \cdot \epsilon_i^j \rceil`.

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

The distance matrix has two components: a Euclidean distance term and a feature-encoded term. Coordinates are drawn from a mixture of Gaussian :math:`\mathcal{N}(0, I)` and uniform :math:`\textbf{U}(-2, 2)` distributions. The feature-encoded component is :math:`\frac{1}{{3}^{deg - 1}} (\frac{1}{\sqrt{p}} (\mathcal{B} x_i)_j + 3)^{deg} \cdot \epsilon_i`, where the elements of :math:`\mathcal{B}` are products of Bernoulli :math:`\textbf{B}(0.5)` and uniform :math:`\textbf{U}(-2, 2)` samples.

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

Let :math:`\bar{r}_{ij} = (\frac{0.05}{\sqrt{p}}(\mathcal{B} \mathbf{x}_i)_j + {0.1}^{\frac{1}{deg}})^{deg}`. The expected return is :math:`\mathbf{r}_i = \bar{\mathbf{r}}_i + \mathbf{L} \mathbf{f} + 0.01 \tau \mathbf{\epsilon}`, and the covariance matrix is :math:`\mathbf{\Sigma} = \mathbf{L} \mathbf{L}^{\intercal} + (0.01 \tau)^2 \mathbf{I}`, where :math:`\mathcal{B}` follows a Bernoulli distribution, :math:`\mathbf{L} \sim \textbf{U}(-0.0025\tau, 0.0025\tau)`, and :math:`\mathbf{f}, \mathbf{\epsilon} \sim \mathcal{N}(0, \mathbf{I})`.

.. autofunction:: pyepo.data.portfolio.genData
    :noindex:

.. code-block:: python

   import pyepo

   num_data = 1000 # number of data
   num_feat = 4 # size of feature
   num_assets = 50 # number of assets
   cov, x, r = pyepo.data.portfolio.genData(num_data, num_feat, num_assets, deg=4, noise_level=1, seed=135)



optDataset
==========

``pyepo.data.optDataset`` is a PyTorch ``Dataset`` that stores features and cost coefficients, and **solves the optimization problem to obtain optimal solutions and objective values**.

``optDataset`` is **not** required for training with PyEPO, but it provides a convenient way to precompute optimal solutions and objective values when they are not available in the original data.

.. autoclass:: pyepo.data.dataset.optDataset
    :noindex:

The following example shows how to use ``optDataset`` with a PyTorch ``DataLoader``:

.. code-block:: python

   import pyepo
   from torch.utils.data import DataLoader

   # model for shortest path
   grid = (5,5) # grid size
   model = pyepo.model.grb.shortestPathModel(grid)

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

``pyepo.data.optDatasetKNN`` is a PyTorch ``Dataset`` for k-nearest neighbors (kNN) robust loss [#f1]_ in decision-focused learning. It stores features and cost coefficients, and computes **mean kNN solutions and optimal objective values**.

.. autoclass:: pyepo.data.dataset.optDatasetKNN
    :noindex:

.. code-block:: python

  import pyepo
  from torch.utils.data import DataLoader

  # model for shortest path
  grid = (5,5) # grid size
  model = pyepo.model.grb.shortestPathModel(grid)

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

.. [#f1] Schutte, N., Postek, K., & Yorke-Smith, N. (2023). Robust Losses for Decision-Focused Learning. arXiv preprint arXiv:2310.04328.
