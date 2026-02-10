Model
+++++

``PyEPO`` supports end-to-end predict-then-optimize with linear objective functions and unknown cost coefficients. At its core is the differentiable optimization solver, which computes gradients of the cost coefficients with respect to the optimal solution.

``optModel`` is the base abstraction in ``PyEPO``. It wraps an optimization solver or algorithm as a container, providing a unified interface for training and evaluation. ``PyEPO`` provides several pre-defined models using GurobiPy, Pyomo, COPT, and MPAX:

* **Shortest path** (GurobiPy, Pyomo, COPT & MPAX)
* **Knapsack** (GurobiPy, Pyomo, COPT & MPAX)
* **Traveling salesman** (GurobiPy, Pyomo & COPT)
* **Portfolio optimization** (GurobiPy, Pyomo & COPT)

When building models with ``PyEPO``, users do **not** need to specify the cost coefficients, since they are unknown and will be predicted from data.

For more details, see the `01 Optimization Model <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/01%20Optimization%20Model.ipynb>`_ notebook.


User-defined Models
===================

Users can define custom optimization problems with linear objective functions. ``PyEPO`` provides three ways to do this:

1. **GurobiPy-based**: Inherit from ``optGrbModel`` and implement ``_getModel``.
2. **Pyomo-based**: Inherit from ``optOmoModel`` and implement ``_getModel``.
3. **From scratch**: Inherit from ``optModel`` and implement ``_getModel``, ``setObj``, ``solve``, and ``num_cost``.

The ``optModel`` interface consists of:

* ``_getModel``: Build and return the optimization model and decision variables.
* ``setObj``: Set the objective function with a given cost vector.
* ``solve``: Solve the problem and return the optimal solution and objective value.


User-defined GurobiPy Models
----------------------------

To define a GurobiPy model, inherit from ``pyepo.model.grb.optGrbModel`` and implement the ``_getModel`` method. The model sense (minimize/maximize) is automatically detected from the GurobiPy model.

.. autoclass:: pyepo.model.grb.optGrbModel
    :noindex:
    :members: __init__, _getModel, setObj, solve, num_cost, relax


For example, consider the following binary optimization problem:

.. math::
  \begin{aligned}
  \max_{x} & \sum_{i=0}^4 c_i x_i \\
  s.t. \quad & 3 x_0 + 4 x_1 + 3 x_2 + 6 x_3 + 4 x_4 \leq 12 \\
  & 4 x_0 + 5 x_1 + 2 x_2 + 3 x_3 + 5 x_4 \leq 10 \\
  & 5 x_0 + 4 x_1 + 6 x_2 + 2 x_3 + 3 x_4 \leq 15 \\
  & \forall x_i \in \{0, 1\}
  \end{aligned}

Users only need to implement the ``_getModel`` method:

.. code-block:: python

   import random

   import gurobipy as gp
   from gurobipy import GRB

   from pyepo.model.grb import optGrbModel

   class myModel(optGrbModel):

       def _getModel(self):
           # create a model
           m = gp.Model()
           # variables
           x = m.addVars(5, name="x", vtype=GRB.BINARY)
           # model sense
           m.modelSense = GRB.MAXIMIZE
           # constraints
           m.addConstr(3 * x[0] + 4 * x[1] + 3 * x[2] + 6 * x[3] + 4 * x[4] <= 12)
           m.addConstr(4 * x[0] + 5 * x[1] + 2 * x[2] + 3 * x[3] + 5 * x[4] <= 10)
           m.addConstr(5 * x[0] + 4 * x[1] + 6 * x[2] + 2 * x[3] + 3 * x[4] <= 15)
           return m, x

   myoptmodel = myModel()
   cost = [random.random() for _ in range(myoptmodel.num_cost)] # random cost vector
   myoptmodel.setObj(cost) # set objective function
   myoptmodel.solve() # solve


User-defined Pyomo Models
-------------------------

To define a Pyomo model, inherit from ``pyepo.model.omo.optOmoModel`` and implement the ``_getModel`` method.

.. autoclass:: pyepo.model.omo.optOmoModel
    :noindex:
    :members: __init__, _getModel, setObj, solve, num_cost, relax

.. warning::  Unlike ``optGrbModel``, ``optOmoModel`` requires explicitly setting ``modelSense`` in ``_getModel``.

Here is the same problem implemented with Pyomo:

.. code-block:: python

   import random

   from pyomo import environ as pe

   from pyepo.model.omo import optOmoModel
   from pyepo import EPO

   class myModel(optOmoModel):

       def _getModel(self):
           # sense
           self.modelSense = EPO.MAXIMIZE
           # create a model
           m = pe.ConcreteModel()
           # variables
           x = pe.Var([0,1,2,3,4], domain=pe.Binary)
           m.x = x
           # constraints
           m.cons = pe.ConstraintList()
           m.cons.add(3 * x[0] + 4 * x[1] + 3 * x[2] + 6 * x[3] + 4 * x[4] <= 12)
           m.cons.add(4 * x[0] + 5 * x[1] + 2 * x[2] + 3 * x[3] + 5 * x[4] <= 10)
           m.cons.add(5 * x[0] + 4 * x[1] + 6 * x[2] + 2 * x[3] + 3 * x[4] <= 15)
           return m, x

   myoptmodel = myModel(solver="gurobi")
   cost = [random.random() for _ in range(myoptmodel.num_cost)] # random cost vector
   myoptmodel.setObj(cost) # set objective function
   myoptmodel.solve() # solve


User-defined Models from Scratch
--------------------------------

For complete flexibility, ``pyepo.model.opt.optModel`` allows users to build models with any solver or algorithm. Override ``_getModel``, ``setObj``, ``solve``, and ``num_cost`` to integrate a custom solver.

.. autoclass:: pyepo.model.opt.optModel
    :noindex:
    :members: __init__, _getModel, setObj, solve, num_cost

.. warning::  ``optModel`` requires setting ``modelSense`` in ``_getModel``. If not set, the default is minimization.

The following example uses ``networkx`` with the Dijkstra algorithm to solve a shortest path problem:

.. code-block:: python

   import random

   import numpy as np
   import networkx as nx

   from pyepo.model.opt import optModel

   class myShortestPathModel(optModel):

       def __init__(self, grid):
           """
           Args:
               grid (tuple): size of grid network
           """
           self.grid = grid
           self.arcs = self._getArcs()
           super().__init__()

       def _getArcs(self):
           """
           A method to get list of arcs for grid network

           Returns:
               list: arcs
           """
           arcs = []
           for i in range(self.grid[0]):
               # edges on rows
               for j in range(self.grid[1] - 1):
                   v = i * self.grid[1] + j
                   arcs.append((v, v + 1))
               # edges in columns
               if i == self.grid[0] - 1:
                   continue
               for j in range(self.grid[1]):
                   v = i * self.grid[1] + j
                   arcs.append((v, v + self.grid[1]))
           return arcs

       def _getModel(self):
           """
           A method to build model

           Returns:
               tuple: optimization model and variables
           """
           # build graph as optimization model
           g = nx.Graph()
           # add arcs as variables
           g.add_edges_from(self.arcs, cost=0)
           return g, g.edges

       def setObj(self, c):
           """
           A method to set objective function

           Args:
               c (ndarray): cost of objective function
           """
           for i, e in enumerate(self.arcs):
               self._model.edges[e]["cost"] = c[i]

       def solve(self):
           """
           A method to solve model

           Returns:
               tuple: optimal solution (list) and objective value (float)
           """
           # dijkstra
           path = nx.shortest_path(self._model, weight="cost", source=0, target=self.grid[0]*self.grid[1]-1)
           # convert path into active edges
           edges = []
           u = 0
           for v in path[1:]:
               edges.append((u,v))
               u = v
           # init sol & obj
           sol = np.zeros(self.num_cost)
           obj = 0
           # convert active edges into solution and obj
           for i, e in enumerate(self.arcs):
               if e in edges:
                   sol[i] = 1 # active edge
                   obj += self._model.edges[e]["cost"] # cost of active edge
           return sol, obj

   # solve model
   grid = (5,5)
   myoptmodel = myShortestPathModel(grid)
   cost = [random.random() for _ in range(myoptmodel.num_cost)] # random cost vector
   myoptmodel.setObj(cost) # set objective function
   sol, obj = myoptmodel.solve() # solve
   # print res
   print('Obj: {}'.format(obj))
   for i, e in enumerate(myoptmodel.arcs):
       if sol[i] > 1e-3:
           print(e)


MPAX Models
===========

MPAX (Mathematical Programming in JAX) is a hardware-accelerated mathematical programming framework based on the PDHG (Primal-Dual Hybrid Gradient) algorithm, designed for large-scale LP problems.

``optMpaxModel`` is a ``PyEPO`` model that uses MPAX to solve LP relaxations via PDHG. It accepts constraints in matrix/vector form:

   - ``A``, ``b``: Equality constraints :math:`Ax = b`. Omit if there are no equality constraints.
   - ``G``, ``h``: Inequality constraints :math:`Gx \leq h`. Omit if there are no inequality constraints.
   - ``l``: Lower bounds (default: 0, i.e., non-negative variables).
   - ``u``: Upper bounds (default: infinity, i.e., unbounded).
   - ``use_sparse_matrix`` (default: ``True``): Whether to use sparse matrix storage.
   - ``minimize`` (default: ``True``): Whether to minimize the objective.

.. autoclass:: pyepo.model.mpax.optMpaxModel
  :noindex:
  :members: __init__, _getModel, setObj, solve, num_cost, relax

.. code-block:: python

   from pyepo.model.mpax import optMpaxModel
   optmodel = optMpaxModel(A=A, b=b, G=G, h=h, use_sparse_matrix=False, minimize=True)

   optmodel.setObj(cost) # set objective function
   optmodel.solve() # solve


Pre-defined Models
==================

``PyEPO`` includes pre-defined models for several classic optimization problems.


Shortest Path
-------------

The shortest path problem finds the minimum-cost path from the northwest corner to the southeast corner of an (h, w) grid network. The default grid size is (5, 5).

.. image:: ../../images/shortestpath.png
  :width: 300
  :alt: Shortest Path on the Grid Graph

The problem is formulated as a minimum cost flow linear program.

Shortest Path GurobiPy Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyepo.model.grb.shortestPathModel
    :noindex:
    :members: __init__, setObj, solve, num_cost

.. code-block:: python

   import pyepo

   grid = (5,5) # network grid
   optmodel = pyepo.model.grb.shortestPathModel(grid) # build model

The ``setObj`` and ``solve`` methods can be called manually, but they are invoked automatically during training.

.. code-block:: python

   import random
   cost = [random.random() for _ in range(optmodel.num_cost)] # random cost vector
   optmodel.setObj(cost) # set objective function
   optmodel.solve() # solve

Shortest Path Pyomo Model
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyepo.model.omo.shortestPathModel
    :noindex:
    :members: __init__, setObj, solve, num_cost

Pyomo supports multiple backend solvers (e.g., BARON, CBC, CPLEX, Gurobi). Specify the solver via the ``solver`` parameter:

.. code-block:: python

   import pyepo

   grid = (5,5) # network grid
   optmodel = pyepo.model.omo.shortestPathModel(grid, solver="glpk") # build model with GLPK
   optmodel = pyepo.model.omo.shortestPathModel(grid, solver="gurobi") # build model with Gurobi

To list available solvers:

.. code-block:: bash

   pyomo help --solvers


Knapsack
--------

The multi-dimensional knapsack problem is a maximization problem: select a subset of items such that total weight does not exceed resource capacities and total value is maximized. Consider a 3-dimensional example:

.. math::
  \begin{aligned}
  \max_{x} & \sum_{i=0}^4 c_i x_i \\
  s.t. \quad & 3 x_0 + 4 x_1 + 3 x_2 + 6 x_3 + 4 x_4 \leq 12 \\
  & 4 x_0 + 5 x_1 + 2 x_2 + 3 x_3 + 5 x_4 \leq 10 \\
  & 5 x_0 + 4 x_1 + 6 x_2 + 2 x_3 + 3 x_4 \leq 15 \\
  & \forall x_i \in \{0, 1\}
  \end{aligned}

The constraint coefficients **weights** and right-hand sides **capacities** define the problem.

.. note:: The number of dimensions and items are determined by the shape of **weights** and **capacities**.

Knapsack GurobiPy Model
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyepo.model.grb.knapsackModel
    :noindex:
    :members: __init__, setObj, solve, num_cost, relax

.. code-block:: python

   import pyepo

   weights = [[3, 4, 3, 6, 4],
              [4, 5, 2, 3, 5],
              [5, 4, 6, 2, 3]] # constraints coefficients
   capacities = [12, 10, 15] # constraints rhs
   optmodel = pyepo.model.grb.knapsackModel(weights, capacities) # build model

.. code-block:: python

   import random
   cost = [random.random() for _ in range(optmodel.num_cost)] # random cost vector
   optmodel.setObj(cost) # set objective function
   optmodel.solve() # solve

The ``relax`` method returns an LP relaxation by removing integrality constraints:

.. code-block:: python

   optmodel_rel = optmodel.relax() # relax

Knapsack Pyomo Model
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyepo.model.omo.knapsackModel
    :noindex:
    :members: __init__, setObj, solve, num_cost, relax

.. code-block:: python

   import pyepo

   weights = [[3, 4, 3, 6, 4],
              [4, 5, 2, 3, 5],
              [5, 4, 6, 2, 3]] # constraints coefficients
   capacities = [12, 10, 15] # constraints rhs
   # build model with GLPK
   optmodel = pyepo.model.omo.knapsackModel(weights, capacities, solver="glpk")
   # build model with Gurobi
   optmodel = pyepo.model.omo.knapsackModel(weights, capacities, solver="gurobi")


Traveling Salesman
------------------

The traveling salesman problem (TSP) seeks the shortest route that visits each city exactly once and returns to the origin. We consider the symmetric TSP with 20 nodes.

Three ILP formulations are available: Dantzig-Fulkerson-Johnson (DFJ), Gavish-Graves (GG), and Miller-Tucker-Zemlin (MTZ).

.. note:: The DFJ formulation uses lazy constraints and is available with GurobiPy and COPT. The GG and MTZ formulations are available with GurobiPy, Pyomo, and COPT.

TSP GurobiPy Models
^^^^^^^^^^^^^^^^^^^^

DFJ formulation
^^^^^^^^^^^^^^^

.. autoclass:: pyepo.model.grb.tspDFJModel
    :noindex:
    :members: __init__, setObj, solve, num_cost

The DFJ formulation has exponentially many subtour elimination constraints, solved via column generation. LP relaxation is **not** supported.

.. code-block:: python

   import pyepo
   import random

   num_nodes = 20 # number of nodes
   optmodel = pyepo.model.grb.tspDFJModel(num_nodes) # build model

   cost = [random.random() for _ in range(optmodel.num_cost)] # random cost vector
   optmodel.setObj(cost) # set objective function
   optmodel.solve() # solve


GG formulation
^^^^^^^^^^^^^^

.. autoclass:: pyepo.model.grb.tspGGModel
    :noindex:
    :members: __init__, setObj, solve, num_cost, relax

.. code-block:: python

   import pyepo
   import random

   num_nodes = 20 # number of nodes
   optmodel = pyepo.model.grb.tspGGModel(num_nodes) # build model

   cost = [random.random() for _ in range(optmodel.num_cost)] # random cost vector
   optmodel.setObj(cost) # set objective function
   optmodel.solve() # solve

   optmodel.relax() # relax


MTZ formulation
^^^^^^^^^^^^^^^

.. autoclass:: pyepo.model.grb.tspMTZModel
    :noindex:
    :members: __init__, setObj, solve, num_cost, relax

.. code-block:: python

   import pyepo
   import random

   num_nodes = 20 # number of nodes
   optmodel = pyepo.model.grb.tspMTZModel(num_nodes) # build model

   cost = [random.random() for _ in range(optmodel.num_cost)] # random cost vector
   optmodel.setObj(cost) # set objective function
   optmodel.solve() # solve

   optmodel.relax() # relax


TSP Pyomo Models
^^^^^^^^^^^^^^^^

The GG and MTZ formulations are available with Pyomo. The DFJ formulation is not supported in Pyomo due to the lack of a native callback API.

.. autoclass:: pyepo.model.omo.tspGGModel
    :noindex:
    :members: __init__, setObj, solve, num_cost, relax

.. autoclass:: pyepo.model.omo.tspMTZModel
    :noindex:
    :members: __init__, setObj, solve, num_cost, relax

.. code-block:: python

   import pyepo
   import random

   num_nodes = 20 # number of nodes

   # GG formulation with Gurobi solver
   optmodel = pyepo.model.omo.tspGGModel(num_nodes, solver="gurobi")
   # MTZ formulation with GLPK solver
   optmodel = pyepo.model.omo.tspMTZModel(num_nodes, solver="glpk")

   cost = [random.random() for _ in range(optmodel.num_cost)] # random cost vector
   optmodel.setObj(cost) # set objective function
   optmodel.solve() # solve


TSP COPT Models
^^^^^^^^^^^^^^^

All three formulations (DFJ, GG, MTZ) are available with COPT. The DFJ formulation uses COPT's callback API for lazy subtour elimination constraints.

.. autoclass:: pyepo.model.copt.tspDFJModel
    :noindex:
    :members: __init__, setObj, solve, num_cost

.. autoclass:: pyepo.model.copt.tspGGModel
    :noindex:
    :members: __init__, setObj, solve, num_cost, relax

.. autoclass:: pyepo.model.copt.tspMTZModel
    :noindex:
    :members: __init__, setObj, solve, num_cost, relax

.. code-block:: python

   import pyepo
   import random

   num_nodes = 20 # number of nodes

   # DFJ formulation
   optmodel = pyepo.model.copt.tspDFJModel(num_nodes)
   # GG formulation
   optmodel = pyepo.model.copt.tspGGModel(num_nodes)
   # MTZ formulation
   optmodel = pyepo.model.copt.tspMTZModel(num_nodes)

   cost = [random.random() for _ in range(optmodel.num_cost)] # random cost vector
   optmodel.setObj(cost) # set objective function
   optmodel.solve() # solve


Portfolio
---------

Portfolio optimization selects an asset allocation that maximizes expected return for a given level of risk:

.. math::
  \begin{aligned}
  \max_{x} & \sum_{i} r_i x_i \\
  s.t. \quad & \sum_{i} x_i = 1 \\
  & \mathbf{x}^{\intercal} \mathbf{\Sigma} \mathbf{x} \leq \gamma \bar{\Sigma}\\
  & \forall x_i \geq 0
  \end{aligned}


Portfolio GurobiPy Model
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyepo.model.grb.portfolioModel
    :noindex:
    :members: __init__, setObj, solve, num_cost

.. code-block:: python

   import pyepo
   import numpy as np

   m = 50 # number of assets
   cov = np.cov(np.random.randn(10, m), rowvar=False) # covariance matrix
   optmodel = pyepo.model.grb.portfolioModel(m, cov) # build model

.. code-block:: python

   import random
   revenue = [random.random() for _ in range(optmodel.num_cost)] # random cost vector
   optmodel.setObj(revenue) # set objective function
   optmodel.solve() # solve

Portfolio Pyomo Model
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyepo.model.omo.portfolioModel
    :noindex:
    :members: __init__, setObj, solve, num_cost

.. code-block:: python

   import pyepo
   import numpy as np

   m = 50 # number of assets
   cov = np.cov(np.random.randn(10, m), rowvar=False) # covariance matrix
   optmodel = pyepo.model.omo.portfolioModel(m, cov, solver="gurobi") # build model

Portfolio COPT Model
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyepo.model.copt.portfolioModel
    :noindex:
    :members: __init__, setObj, solve, num_cost

.. code-block:: python

   import pyepo
   import numpy as np

   m = 50 # number of assets
   cov = np.cov(np.random.randn(10, m), rowvar=False) # covariance matrix
   optmodel = pyepo.model.copt.portfolioModel(m, cov) # build model
