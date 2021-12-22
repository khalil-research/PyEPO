Data
++++

``spo.data`` contains synthetic data generator and a dataset class ``optDataset`` to wrap data samples.


Data Generator
==============

``spo.data`` includes data generators for the shortest path, multi-dimensional knapsack, and traveling salesman.

The generated data include input features :math:`x` and objective function coefficients :math:`c`. Generally, we create synthetic dataset :math:`\mathcal{D} = \{(x_1, c_1), (x_2, c_2), ..., (x_n, c_n)\} \in \mathbb{R}^{n \times p}`. The feature vector :math:`x_i \in \mathbb{R}^p` follows a standard multivariate Gaussian distribution :math:`\mathcal{N}(0, I_p)`. A function :math:`f(\cdot)` with random noise encodes the features :math:`x_i` into the objective function coefficients :math:`c_i`.

Here, ``deg`` is a positive integer parameter for polynomial degree and ``noise_width`` :math:`\epsilon_i^j \sim  U(1-\bar{\epsilon}_i^j, 1+\bar{\epsilon}_i^j)` is a multiplicative noise term.


Shortest Path
-------------

For the shortest path, a random matrix :math:`\mathcal{B} \in \mathbb{R}^{d \times p}` which follows Bernoulli distribution with probability :math:`0.5`, encode the features :math:`x_i`. The cost of objective function :math:`c_{ij}` is generated from :math:`[\frac{1}{\sqrt{p}} ((\mathcal{B} x_i)_j + 3)^{deg} + 1] \cdot \epsilon_i^j`.

.. autofunction:: spo.data.shortestpath.genData

The following code is to generate data for the shortest path on the grid network:

.. code-block:: python

   import spo

   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   grid = (5,5) # grid size
   x, c = spo.data.shortestpath.genData(num_data, num_feat, grid, deg=4, noise_width=0, seed=135)


Knapsack
--------

For the knapsack, the weights of items are fixed for the data and are randomly sampled from :math:`3` to :math:`8` with a precision of :math:`1` decimal place. Same as the shortest path, :math:`\mathcal{B} \in \mathbb{R}^{d \times p}` is a random matrix in which all elements follow Bernoulli distribution with probability :math:`0.5`. Then, the objective function coeficients are define as :math:`c_{ij} = \lceil \frac{5}{{3.5}^{deg}} (\frac{1}{\sqrt{p}} ((\mathcal{B} x_i)_j + 3)^{deg} + 1) \cdot \epsilon_i^j \rceil`.

.. autofunction:: spo.data.knapsack.genData

The following code is to generate data for the 3d-knapsack:

.. code-block:: python

   import spo

   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   num_item = 32 # number of items
   dim = 3 # dimension of knapsack
   weights, x, c = spo.data.knapsack.genData(num_data, num_feat, num_item, dim, deg=4, noise_width=0, seed=135)


Traveling Salesman
------------------

The distance consists of two parts: one comes from Euclidean distance, the other derived from feature encoding. For Euclidean distance, we create coordinates from the mixture of Gaussian distribution :math:`\mathcal{N}(0, I)` and uniform distribution :math:`\textbf{U}(-2, 2)`. For feature encoding, it is :math:`\frac{1}{{3}^{deg - 1} \sqrt{p}} ((\mathcal{B} x_i)_j + 3)^{deg} \cdot \epsilon_i`, where the elements of :math:`\mathcal{B}` come from the multiplication of Bernoulli :math:`\textbf{B}(0.5)` and uniform :math:`\textbf{U}(-2, 2)`.

.. autofunction:: spo.data.tsp.genData

The following code is to generate data for the tsp:

.. code-block:: python

   import spo

   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   num_node = 20 # number of nodes
   x, c = spo.data.tsp.genData(num_data, num_feat, num_node, deg=4, noise_width=0, seed=135)


optDataset
==========

``spo.data.optDataset`` is PyTorch Dataset, which stores the features and their corresponding costs of the objective function and solves optimization problems to get optimal solutions and optimal objective values.

.. autoclass:: spo.data.dataset.optDataset
   :members: __init__

As the following example, ``optDataset`` and Pytorch ``DataLoader`` wrap the data samples, which can make the model training cleaner and more organized.

.. code-block:: python

   import spo
   from torch.utils.data import DataLoader

   # model for shortest path
   grid = (5,5) # grid size
   model = spo.model.grb.shortestPathModel(grid)

   # generate data
   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   deg = 4 # polynomial degree
   noise_width = 0 # noise width
   x, c = spo.data.shortestpath.genData(num_data, num_feat, grid, deg, noise_width, seed=135)

   # build dataset
   dataset = spo.data.dataset.optDataset(model, x, c)

   # get data loader
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
