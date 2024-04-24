Data
++++

``pyepo.data`` contains synthetic data generator and a dataset class ``optDataset`` to wrap data samples.

For more information and details about the Dataset, please see the `02 Optimization Dataset <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/03%20Training%20and%20Testing.ipynb>`_


Data Generator
==============

``pyepo.data`` includes synthetic datasets for three of the most classic optimization problems: the shortest path problem, the multi-dimensional knapsack problem, the traveling salesperson problem, and portfolio optimization.

The synthetic datasets include features :math:`\mathbf{x}` and cost coefficients :math:`\mathbf{c}`. The feature vector :math:`\mathbf{x}_i \in \mathbb{R}^p` follows a standard multivariate Gaussian distribution :math:`\mathcal{N}(0, \mathbf{I})`, and the corresponding cost :math:`\mathbf{c}_i \in \mathbb{R}^d` comes from a polynomial function :math:`f(\mathbf{x}_i)` multiplicated with a random noise :math:`\mathbf{\epsilon}_i \sim  U(1-\bar{\epsilon}, 1+\bar{\epsilon})`.
In general, there are several parameters that users can control:

* **num_data** (:math:`n`): data size

* **num_features** (:math:`p`): feature dimension of costs :math:`\mathbf{c}`

* **deg** (:math:`deg`): polynomial degree of function :math:`f(\mathbf{x}_i)`

* **noise_width** (:math:`\bar{\epsilon}`):  noise half-width of :math:`\mathbf{\epsilon}`

* **seed**: random state seed to generate data


Shortest Path
-------------

For the shortest path, a random matrix :math:`\mathcal{B} \in \mathbb{R}^{d \times p}` which follows Bernoulli distribution with probability :math:`0.5`, encode the features :math:`x_i`. The cost of objective function :math:`c_{ij}` is generated from :math:`c_i^j = [\frac{1}{{3.5}^{deg}} (\frac{1}{\sqrt{p}}(\mathcal{B} \mathbf{x}_i)_j + 3)^{deg} + 1] \cdot \epsilon_i^j`.

.. autofunction:: pyepo.data.shortestpath.genData
    :noindex:

The following code is to generate data for the shortest path on the grid network:

.. code-block:: python

   import pyepo

   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   grid = (5,5) # grid size
   x, c = pyepo.data.shortestpath.genData(num_data, num_feat, grid, deg=4, noise_width=0, seed=135)


Knapsack
--------

Because we assume that the uncertain coefficients exist only on the objective function, the weights of items are fixed throughout the data. We define the number of items as :math:`m` and the dimension of resources is :math:`k`. The weights :math:`\mathcal{W} \in \mathbb{R}^{k \times m}` are sampled from :math:`3` to :math:`8` with a precision of :math:`1` decimal place. With the same :math:`\mathcal{B}`, cost :math:`c_{ij}` is calculated from :math:`c_i^j = \lceil [\frac{5}{{3.5}^{deg}} (\frac{1}{\sqrt{p}}(\mathcal{B} \mathbf{x}_i)_j + 3)^{deg} + 1] \cdot \epsilon_i^j \rceil`.

.. autofunction:: pyepo.data.knapsack.genData
    :noindex:

The following code is to generate data for the 3d-knapsack:

.. code-block:: python

   import pyepo

   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   num_item = 32 # number of items
   dim = 3 # dimension of knapsack
   weights, x, c = pyepo.data.knapsack.genData(num_data, num_feat, num_item, dim, deg=4, noise_width=0, seed=135)


Traveling Salesperson
---------------------

The distance consists of two parts: one comes from Euclidean distance, the other derived from feature encoding. For Euclidean distance, we create coordinates from the mixture of Gaussian distribution :math:`\mathcal{N}(0, I)` and uniform distribution :math:`\textbf{U}(-2, 2)`. For feature encoding, it is :math:`\frac{1}{{3}^{deg - 1}} (\frac{1}{\sqrt{p}} (\mathcal{B} x_i)_j + 3)^{deg} \cdot \epsilon_i`, where the elements of :math:`\mathcal{B}` come from the multiplication of Bernoulli :math:`\textbf{B}(0.5)` and uniform :math:`\textbf{U}(-2, 2)`.

.. autofunction:: pyepo.data.tsp.genData
    :noindex:

The following code is to generate data for the Traveling salesperson:

.. code-block:: python

   import pyepo

   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   num_node = 20 # number of nodes
   x, c = pyepo.data.tsp.genData(num_data, num_feat, num_node, deg=4, noise_width=0, seed=135)


Portfolio
---------

Let :math:`\bar{r}_{ij} = (\frac{0.05}{\sqrt{p}}(\mathcal{B} \mathbf{x}_i)_j + {0.1}^{\frac{1}{deg}})^{deg}`. In the context of portfolio optimization, the expected return of the assets :math:`\mathbf{r}_i` is defined as :math:`\bar{\mathbf{r}}_i + \mathbf{L} \mathbf{f} + 0.01 \tau \mathbf{\epsilon}` and the covariance matrix :math:`\mathbf{\Sigma}` is expressed :math:`\mathbf{L} \mathbf{L}^{\intercal} + (0.01 \tau)^2 \mathbf{I}`, where :math:`\mathcal{B}` follows Bernoulli distribution, :math:`\mathbf{L}` follows uniform distribution between :math:`-0.0025 \tau` and :math:`0.0025 \tau`, and :math:`\mathbf{f}` and :math:`\mathbf{\epsilon}` follow  standard normal distribution.

.. autofunction:: pyepo.data.portfolio.genData
    :noindex:

The following code is to generate data for the portfolio:

.. code-block:: python

   import pyepo

   num_data = 1000 # number of data
   num_feat = 4 # size of feature
   num_assets = 50 # number of assets
   cov, x, r = pyepo.data.portfolio.genData(num_data, num_feat, num_assets, deg=4, noise_level=1, seed=135)



optDataset
==========

``pyepo.data.optDataset`` is PyTorch Dataset, which stores the features and their corresponding costs of the objective function, and **solves optimization problems to get optimal solutions and optimal objective values**.

``optDataset`` is **not** necessary for training with PyEPO, but it can be easier to obtain optimal solutions and objective values when they are not available in the original data.

.. autoclass:: pyepo.data.dataset.optDataset
    :noindex:

As the following example, ``optDataset`` and Pytorch ``DataLoader`` wrap the data samples, which can make the model training cleaner and more organized.

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
