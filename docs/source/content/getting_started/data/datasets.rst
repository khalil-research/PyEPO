Datasets
++++++++

The ``optDataset`` family turns features and costs into a PyTorch ``Dataset`` with the optimization labels the training methods need. Each dataset builds those labels by solving an ``optModel``. The model can be a :doc:`built-in one <../model/builtin>` or one you :doc:`define yourself <../model>`.


optDataset
==========

``pyepo.data.optDataset`` is a PyTorch ``Dataset`` that stores features and cost coefficients, and **solves the optimization problem to obtain optimal solutions and objective values**. The features and costs can come from any arrays. The generators in :doc:`generators` are only a convenience.

``optDataset`` precomputes :math:`\mathbf{w}^*(\mathbf{c})` and :math:`z^*(\mathbf{c})` at construction time. If those labels already exist from another source, you can skip ``optDataset`` and feed ``(x, c, w, z)`` batches to ``pyepo.func`` modules directly.

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
   num_data = 1000 # number of samples
   num_feat = 5 # number of features
   deg = 4 # polynomial degree
   noise_width = 0 # noise width
   x, c = pyepo.data.shortestpath.genData(num_data, num_feat, grid, deg, noise_width, seed=135)

   # build dataset
   dataset = pyepo.data.dataset.optDataset(model, x, c)

   # get data loader
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


optDatasetKNN
=============

``pyepo.data.optDatasetKNN`` is a PyTorch ``Dataset`` that implements the k-nearest neighbors (kNN) robust loss [#f1]_ for predict-then-optimize training. It stores features and cost coefficients, then computes the **mean k-nearest-neighbor solutions and the corresponding optimal objective values**.

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
  num_data = 1000 # number of samples
  num_feat = 5 # number of features
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

``optDatasetConstrs`` currently requires a Gurobi-backed ``optModel``. The dataset also checks that the optimal vertex is binary, since CaVE is defined for binary linear programs. It raises on infeasible instances or non-binary optima unless ``skip_infeas=True``, which drops them instead.

For a runnable walkthrough that uses ``optDatasetConstrs`` end-to-end with the CaVE loss, see the `04 CaVE for Binary Linear Programs <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/04%20CaVE%20for%20Binary%20Linear%20Programs.ipynb>`_ notebook.

.. autoclass:: pyepo.data.dataset.optDatasetConstrs
    :noindex:

Per-instance constraint matrices can have different row counts because different constraints bind at different vertices. Batch them with ``optDataLoader``, which pads them automatically:

.. autoclass:: pyepo.data.dataset.optDataLoader
    :noindex:

.. code-block:: python

  import pyepo
  from pyepo.data.dataset import optDatasetConstrs, optDataLoader

  # model for TSP (Gurobi backend required)
  model = pyepo.model.tspModel(num_nodes=10, formulation="DFJ")

  # generate data
  x, c = pyepo.data.tsp.genData(num_data=1000, num_features=5, num_nodes=10, deg=4, seed=135)

  # build CaVE dataset (extracts tight binding-constraint normals at the optimum)
  dataset_constr = optDatasetConstrs(model, x, c)

  # optDataLoader pads ragged per-instance constraint matrices
  dataloader_constr = optDataLoader(dataset_constr, batch_size=32, shuffle=True)

.. [#f1] Schutte, N., Postek, K., & Yorke-Smith, N. (2024). Robust Losses for Decision-Focused Learning. Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence.
.. [#f2] Tang, B., & Khalil, E. B. (2024). CaVE: A Cone-Aligned Approach for Fast Predict-then-Optimize with Binary Linear Programs. In Integration of Constraint Programming, Artificial Intelligence, and Operations Research (pp. 193-210).
