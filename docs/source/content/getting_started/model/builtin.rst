Built-in Problem Models
+++++++++++++++++++++++

``PyEPO`` includes built-in models for several classic problems. Each model is built by a factory that takes a ``backend`` keyword (default ``"gurobi"``). Pair a model with generated data (:doc:`../data/generators`) and an ``optDataset`` (:doc:`../data/datasets`):

.. code-block:: python

   import pyepo
   from pyepo import model

   grid = (5, 5)
   x, c = pyepo.data.shortestpath.genData(1000, 5, grid, deg=4, seed=135)
   optmodel = model.shortestPathModel(grid)                  # default Gurobi
   dataset = pyepo.data.dataset.optDataset(optmodel, x, c)

Switch the solver with ``backend``. The generic backends take a ``solver=`` argument that names the solver to run:

.. code-block:: python

   model.shortestPathModel(grid, backend="copt")
   model.shortestPathModel(grid, backend="pyomo", solver="glpk")
   model.shortestPathModel(grid, backend="mpax")             # LP on GPU

.. note:: In end-to-end training, ``pyepo.func`` modules call ``setObj`` and ``solve`` during the forward pass.


Shortest Path Model
===================

Minimum-cost path from the northwest to the southeast corner of an ``(h, w)`` grid, formulated as a minimum-cost-flow LP. Supported backends: gurobi, copt, pyomo, ortools, mpax.

.. autofunction:: pyepo.model.shortestPathModel


Knapsack Model
==============

Multi-dimensional 0/1 knapsack: maximize value subject to per-dimension capacities. ``weights`` has shape ``(dim, n_items)`` and ``capacity`` has length ``dim``. Supported backends: gurobi, copt, pyomo, ortools, mpax (LP relaxation).

.. code-block:: python

   weights = [[3, 4, 3, 6, 4], [4, 5, 2, 3, 5], [5, 4, 6, 2, 3]]
   capacity = [12, 10, 15]
   optmodel = model.knapsackModel(weights, capacity)

.. autofunction:: pyepo.model.knapsackModel


Traveling Salesperson Model
===========================

Shortest tour visiting each city once. ``formulation`` is ``"DFJ"`` (lazy subtour elimination), ``"GG"``, or ``"MTZ"``. Supported backends: gurobi and copt (all three formulations), and pyomo (GG, MTZ). On gurobi, ``recycle_cuts=True`` keeps subtour cuts found in one solve for later solves.

.. code-block:: python

   optmodel = model.tspModel(20, formulation="DFJ", recycle_cuts=True)

.. autofunction:: pyepo.model.tspModel


Capacitated Vehicle Routing Model
=================================

Shortest vehicle routes from a depot that serve every customer within capacity. ``formulation`` is ``"RCI"`` (lazy rounded-capacity cuts) or ``"MTZ"``. Supported backends: gurobi and copt (both formulations), and pyomo (MTZ). On gurobi, ``"RCI"`` also accepts ``recycle_cuts=True`` to keep cuts found in one solve for later solves.

.. code-block:: python

   optmodel = model.vrpModel(10, demands=[2, 1, 3, 2, 1, 2, 1, 3, 2],
                             capacity=5.0, num_vehicle=3, formulation="RCI")

.. autofunction:: pyepo.model.vrpModel


Portfolio Model
===============

Mean-variance allocation that maximizes return under a risk budget. Supported backends: gurobi, copt, pyomo.

.. code-block:: python

   import numpy as np
   covariance = np.cov(np.random.randn(1000, 50), rowvar=False)
   optmodel = model.portfolioModel(50, covariance)

.. autofunction:: pyepo.model.portfolioModel
