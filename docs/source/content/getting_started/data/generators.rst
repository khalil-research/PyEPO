Data Generators
+++++++++++++++

``pyepo.data`` includes synthetic data generators for four optimization problems: shortest path, multi-dimensional knapsack, traveling salesperson, and portfolio optimization.

Each generator produces feature-cost pairs :math:`(\mathbf{x}, \mathbf{c})`. The feature vector :math:`\mathbf{x}_i \in \mathbb{R}^p` follows a standard multivariate Gaussian distribution :math:`\mathcal{N}(0, \mathbf{I})`, and the cost :math:`\mathbf{c}_i \in \mathbb{R}^d` is computed from a polynomial function :math:`f(\mathbf{x}_i)` scaled by a multiplicative noise factor :math:`\boldsymbol{\epsilon}_i \sim U(1-\bar{\epsilon}, 1+\bar{\epsilon})`.

Common parameters across the generators:

* **num_data** (:math:`n`): number of data samples

* **num_features** (:math:`p`): feature dimension

* **deg** (:math:`deg`): polynomial degree of the mapping :math:`f(\mathbf{x}_i)`

* **noise_width** (:math:`\bar{\epsilon}`): noise half-width for shortest path, knapsack, and TSP. Portfolio uses ``noise_level`` instead; see below.

* **seed**: random seed for reproducibility


Shortest Path
=============

A random matrix :math:`\mathcal{B} \in \mathbb{R}^{d \times p}` with Bernoulli(0.5) entries maps the feature vector to the cost coefficients: :math:`c_i^j = \tfrac{1}{{3.5}^{deg}} \big[\big(\tfrac{1}{\sqrt{p}}(\mathcal{B} \mathbf{x}_i)_j + 3\big)^{deg} + 1\big] \cdot \epsilon_i^j`.

.. autofunction:: pyepo.data.shortestpath.genData
    :noindex:

.. code-block:: python

   import pyepo

   num_data = 1000 # number of samples
   num_feat = 5 # number of features
   grid = (5,5) # grid size
   x, c = pyepo.data.shortestpath.genData(num_data, num_feat, grid, deg=4, noise_width=0, seed=135)


Knapsack
========

Only the cost coefficients are uncertain; item weights are fixed. Let :math:`m` be the number of items and :math:`k` the number of resource dimensions. The weights :math:`\mathcal{W} \in \mathbb{R}^{k \times m}` are drawn uniformly from the half-open range :math:`[3, 8)` to two decimal places. The cost coefficients are :math:`c_i^j = \big\lceil \tfrac{5}{{3.5}^{deg}} \big[\big(\tfrac{1}{\sqrt{p}}(\mathcal{B} \mathbf{x}_i)_j + 3\big)^{deg} + 1\big] \cdot \epsilon_i^j \big\rceil`.

.. autofunction:: pyepo.data.knapsack.genData
    :noindex:

.. code-block:: python

   import pyepo

   num_data = 1000 # number of samples
   num_feat = 5 # number of features
   num_item = 32 # number of items
   dim = 3 # dimension of knapsack
   weights, x, c = pyepo.data.knapsack.genData(num_data, num_feat, num_item, dim, deg=4, noise_width=0, seed=135)


Traveling Salesperson
=====================

The distance matrix has two components: a Euclidean distance term and a feature-encoded term. Coordinates are drawn from a mixture of a Gaussian distribution :math:`\mathcal{N}(0, \mathbf{I})` and a uniform distribution :math:`U(-2, 2)`. The feature-encoded component is :math:`\tfrac{1}{{3}^{deg - 1}} \big(\tfrac{1}{\sqrt{p}} (\mathcal{B} \mathbf{x}_i)_j + 3\big)^{deg} \cdot \boldsymbol{\epsilon}_i`, where the elements of :math:`\mathcal{B}` are products of Bernoulli(0.5) and uniform :math:`U(-2, 2)` samples.

.. autofunction:: pyepo.data.tsp.genData
    :noindex:

.. code-block:: python

   import pyepo

   num_data = 1000 # number of samples
   num_feat = 5 # number of features
   num_node = 20 # number of nodes
   x, c = pyepo.data.tsp.genData(num_data, num_feat, num_node, deg=4, noise_width=0, seed=135)


Portfolio
=========

Let :math:`\bar{r}_{ij} = \big(\tfrac{0.05}{\sqrt{p}}(\mathcal{B} \mathbf{x}_i)_j + {0.1}^{\frac{1}{deg}}\big)^{deg}`. The expected return is :math:`\mathbf{r}_i = \bar{\mathbf{r}}_i + \mathbf{L} \mathbf{f} + 0.01 \tau \boldsymbol{\epsilon}`, and the covariance matrix is :math:`\mathbf{\Sigma} = \mathbf{L} \mathbf{L}^{\intercal} + (0.01 \tau)^2 \mathbf{I}`, where :math:`\mathcal{B}` has Bernoulli(0.5) entries, :math:`\mathbf{L} \sim U(-0.0025\tau, 0.0025\tau)`, and :math:`\mathbf{f}, \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})`.

Unlike the other generators, portfolio noise is controlled by **noise_level** (:math:`\tau`), which scales both the factor loadings :math:`\mathbf{L}` and the residual noise. ``noise_width`` does not apply here.

.. autofunction:: pyepo.data.portfolio.genData
    :noindex:

.. code-block:: python

   import pyepo

   num_data = 1000 # number of samples
   num_feat = 4 # number of features
   num_assets = 50 # number of assets
   cov, x, r = pyepo.data.portfolio.genData(num_data, num_feat, num_assets, deg=4, noise_level=1, seed=135)
