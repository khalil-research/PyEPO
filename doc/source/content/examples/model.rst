Model
+++++

``PyEPO`` contains several pre-defined optimizations models with GurobiPy and Pyomo. It includes the shortest path problem (GurobiPy & Pyomo), the knapsack problem (GurobiPy & Pyomo), and the traveling salesman problem (GurobiPy).

Our API is also designed to support users to define their own problems based on GurobiPy and Pyomo. Besides the API of GurobiPy & Pyomo, users can also build problems from scratch with whatever solvers and algorithms they want to use.

To build optimizations models with ``PyEPO``, users do **not** need specific costs of objective functions since the cost vector is unknown but can be estimated from data.


Pre-defined Models
==================

Pre-defined models are classic optimization problems, including shortest path, knapsack, and traveling salesman.


Shortest Path
-------------

It is a (h,w) grid network and the goal is to find the shortest path from northwest to southeast. In our examples, the grid size of network is (5,5).

.. image:: ../../images/shortestpath.png
  :width: 300
  :alt: Shortest Path on the Grid Graph

The shortest path problem is built as Linear Programming (LP) and formulated as a minimum cost flow problem. Thus, network flow constraints are modeled as the feasible region.

Shortest Path GurobiPy Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``optModel`` is built from ``pyepo.model.grb.shortestPathModel``, in which API uses GurobiPy to model the shortest path problem.

.. autoclass:: pyepo.model.grb.shortestPathModel
   :members: __init__, setObj, solve, num_cost

.. code-block:: python

   import pyepo

   grid = (5,5) # network grid
   optmodel = pyepo.model.grb.shortestPathModel(grid) # build model

Users can use ``setObj`` with a specific cost vector to set current objective function and use ``solve`` to optimize.

.. code-block:: python

   import random
   cost = [random.random() for _ in range(optmodel.num_cost)] # random cost vector
   optmodel.setObj(cost) # set objective function
   optmodel.solve() # solve

Shortest Path Pyomo Model
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``optModel`` is built from ``pyepo.model.omo.shortestPathModel``, in which API uses Pyomo to model the shortest path problem.

.. autoclass:: pyepo.model.omo.shortestPathModel
   :members: __init__, setObj, solve, num_cost

Pyomo supports a wide variety of solvers in the background (e.g. BARON, CBC, CPLEX, and Gurobi). ``pyepo.model.omo.shortestPathModel`` support users to call different solvers with class parameter ``solver``.

.. code-block:: python

   import pyepo

   grid = (5,5) # network grid
   optmodel = pyepo.model.omo.shortestPathModel(grid, solver="glpk") # build model with glpk
   optmodel = pyepo.model.omo.shortestPathModel(grid, solver="gurobi") # build model with gurobi

You can get the current list of supported solvers using the pyomo command:

.. code-block:: bash

   pyomo help --solvers

Same as ``pyepo.model.grb.shortestPathModel``, methods ``setObj`` and ``solve`` can specify objective function and solve the problem.

.. code-block:: python

   import random
   cost = [random.random() for _ in range(optmodel.num_cost)] # random cost vector
   optmodel.setObj(cost) # set objective function
   optmodel.solve() # solve


Knapsack
--------

Multi-dimensional knapsack problem is a knapsack problem with multiple resource constraints: Given a set of items, the aim is to find a collection that the total weights in is less than or equal to resource capacities and the total value is as large as possible. Let's define a 3d knapsack problem as follow:

.. math::
  \begin{aligned}
  \max_{x} & \sum_{i=0}^4 c_i x_i \\
  s.t. \quad & 3 x_0 + 4 x_1 + 3 x_2 + 6 x_3 + 4 x_4 \leq 12 \\
  & 4 x_0 + 5 x_1 + 2 x_2 + 3 x_3 + 5 x_4 \leq 10 \\
  & 5 x_0 + 4 x_1 + 6 x_2 + 2 x_3 + 3 x_4 \leq 15 \\
  & \forall x_i \in \{0, 1\}
  \end{aligned}

Constraints coefficients **weights** and constraints rhs **capacities** are required to create models.

.. note:: The dimension of the knapsack and the number of items are implicitly defined by the shape of **weights** and **capacities**.

Knapsack GurobiPy Model
^^^^^^^^^^^^^^^^^^^^^^^

The ``optModel`` is built from ``pyepo.model.grb.knapsackModel``, in which API uses GurobiPy to model the knapsack problem.

.. autoclass:: pyepo.model.grb.knapsackModel
   :members: __init__, setObj, solve, num_cost, relax

.. code-block:: python

   import pyepo

   weights = [[3, 4, 3, 6, 4],
              [4, 5, 2, 3, 5],
              [5, 4, 6, 2, 3]] # constraints coefficients
   capacities = [12, 10, 15] # constraints rhs
   optmodel = pyepo.model.grb.knapsackModel(weights, capacities) # build model

Users can use ``setObj`` with a specific cost vector to set current objective function and use ``solve`` to solve it.

.. code-block:: python

   import random
   cost = [random.random() for _ in range(optmodel.num_cost)] # random cost vector
   optmodel.setObj(cost) # set objective function
   optmodel.solve() # solve

In mathematics, the relaxation of a (Mixed) Integer Linear Programming is the problem that arises by removing the integrality constraint of each variable. As an ILP, ``optmodel`` allows users to relax ILP with ``relax`` method to obtain a new relaxed ``optModel``.

.. code-block:: python

   optmodel_rel = optmodel.relax() # relax

Knapsack Pyomo Model
^^^^^^^^^^^^^^^^^^^^

The ``optModel`` is built from ``pyepo.model.omo.knapsackModel``, in which API uses Pyomo to model the knapsack problem.

.. autoclass:: pyepo.model.omo.knapsackModel
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

Same as ``pyepo.model.grb.knapsackModel``,  users can use ``setObj``, ``solve``, and ``relax`` methods.

You can get the current list of supported solvers using the pyomo command:

.. code-block:: bash

   pyomo help --solvers


Traveling Salesman
------------------

The traveling salesman problem (TSP) is the shortest route that visits each city exactly once and returns to the origin city. We consider the symmetric TSP, in which the distance between two cities is the same in each opposite direction. In our examples, the number of nodes is 20.

The TSP can be formulated as an Integer Linear Programming with several formulations. We implemented Dantzig–Fulkerson–Johnson (DFJ) formulation, Gavish–Graves (GG) formulation, and Miller–Tucker–Zemlin (MTZ) formulation.

.. note:: The implementation of TSP is only based on GurobiPy. Pyomo is not supported.

DFJ formulation
^^^^^^^^^^^^^^^

.. autoclass:: pyepo.model.grb.tspDFJModel
   :members: __init__, setObj, solve, num_cost

The number of subtour elimination constraints for DFJ formulation is exponential. Thus, we solved it with column generation. Because of that, the linear relaxation of DFJ is **not** supported in our implementation.

Same as previous model, the code for traveling salesman problem with DFJ formulation is as follows:

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
   :members: __init__, setObj, solve, num_cost, relax

Same as previous model, the code for traveling salesman problem with GG formulation is as follows:

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
   :members: __init__, setObj, solve, num_cost, relax

Same as previous model, the code for traveling salesman problem with MTZ formulation is as follows:

.. code-block:: python

   import pyepo
   import random

   num_nodes = 20 # number of nodes
   optmodel = pyepo.model.grb.tspMTZModel(num_nodes) # build model

   cost = [random.random() for _ in range(optmodel.num_cost)] # random cost vector
   optmodel.setObj(cost) # set objective function
   optmodel.solve() # solve

   optmodel.relax() # relax


User-defined Models
===================

User can build optimization problem with linear objective function.


User-defined GurobiPy Models
----------------------------

User-defined models with GurobiPy can be easily defined by the inheritance of the abstract class ``pyepo.model.grb.optGrbModel``.

For ``optGrbModel``, users does not need specify the sense of optimization model. Both the minimization and maximization models are correctly recognized and run by ``pyepo``.

.. autoclass:: pyepo.model.grb.optGrbModel
   :members: __init__, _getModel, setObj, solve, num_cost, relax


For example, users can build models for the following problem:

.. math::
  \begin{aligned}
  \max_{x} & \sum_{i=0}^4 c_i x_i \\
  s.t. \quad & 3 x_0 + 4 x_1 + 3 x_2 + 6 x_3 + 4 x_4 \leq 12 \\
  & 4 x_0 + 5 x_1 + 2 x_2 + 3 x_3 + 5 x_4 \leq 10 \\
  & 5 x_0 + 4 x_1 + 6 x_2 + 2 x_3 + 3 x_4 \leq 15 \\
  & \forall x_i \in \{0, 1\}
  \end{aligned}

In the general case, users only need to implement ``_getModel`` method with GurobiPy.

.. code-block:: python

   import random

   import gurobipy as gp
   from gurobipy import GRB

   from pyepo.model.grb import optGrbModel

   class myModel(optGrbModel):

       def _getModel(self):
           # ceate a model
           m = gp.Model()
           # varibles
           x = m.addVars(5, name="x", vtype=GRB.BINARY)
           # sense (must be minimize)
           m.modelSense = GRB.MAXIMIZE
           # constraints
           m.addConstr(3 * x[0] + 4 * x[1] + 3 * x[2] + 6 * x[3] + 4 * x[4] <= 12)
           m.addConstr(4 * x[0] + 5 * x[1] + 2 * x[2] + 3 * x[3] + 5 * x[4] <= 10)
           m.addConstr(5 * x[0] + 4 * x[1] + 6 * x[2] + 2 * x[3] + 3 * x[4] <= 15)
           return m, x

   myoptmodel = myModel()
   cost = [random.random() for _ in range(model.num_cost)] # random cost vector
   myoptmodel.setObj(cost) # set objective function
   myoptmodel.solve() # solve


User-defined Pyomo Models
-------------------------

User-defined models with Pyomo can be easily defined by the inheritance of the abstract class ``pyepo.model.omo.optOmoModel``.

.. autoclass:: pyepo.model.omo.optOmoModel
   :members: __init__, _getModel, setObj, solve, num_cost, relax


Let's build models for the problem again with Pyomo:

.. math::
  \begin{aligned}
  \max_{x} & \sum_{i=0}^4 c_i x_i \\
  s.t. \quad & 3 x_0 + 4 x_1 + 3 x_2 + 6 x_3 + 4 x_4 \leq 12 \\
  & 4 x_0 + 5 x_1 + 2 x_2 + 3 x_3 + 5 x_4 \leq 10 \\
  & 5 x_0 + 4 x_1 + 6 x_2 + 2 x_3 + 3 x_4 \leq 15 \\
  & \forall x_i \in \{0, 1\}
  \end{aligned}

In the general case, users only need to implement ``_getModel`` method with Pyomo.

.. warning::  Unlike the ``optGrbModel``, the ``optOmoModel`` need to set ``modelSense`` in the ``_getModel``.

.. code-block:: python

   import random

   from pyomo import environ as pe

   from pyepo.model.omo import optOmoModel
   from pyepo import EPO

   class myModel(optOmoModel):

       def _getModel(self):
           # sense
           self.modelSense = EPO.MAXIMIZE
           # ceate a model
           m = pe.ConcreteModel()
           # varibles
           x = pe.Var([0,1,2,3,4], domain=pe.Binary)
           m.x = x
           # constraints
           m.cons = pe.ConstraintList()
           m.cons.add(3 * x[0] + 4 * x[1] + 3 * x[2] + 6 * x[3] + 4 * x[4] <= 12)
           m.cons.add(4 * x[0] + 5 * x[1] + 2 * x[2] + 3 * x[3] + 5 * x[4] <= 10)
           m.cons.add(5 * x[0] + 4 * x[1] + 6 * x[2] + 2 * x[3] + 3 * x[4] <= 15)
           return m, x

   model = myModel()
   cost = [random.random() for _ in range(model.num_cost)] # random cost vector
   model.setObj(cost) # set objective function
   model.solve() # solve


User-defined Models from Scratch
--------------------------------

``pyepo.model.opt.optModel`` provides an abstract class for users to create an optimization model with any solvers or algorithms. By overriding ``_getModel``, ``setObj``, ``solve``,  and ``num_cost``, user-defined optModel can work for SPO+ and differebntiable Black-box optimizer.

.. autoclass:: pyepo.model.opt.optModel
   :members: __init__, _getModel, setObj, solve, num_cost

.. warning::  The ``optModel`` need to set ``modelSense`` in the ``_getModel``. If not set, the default is to minimize.

For example, we can use ``networkx`` to solve the previous shortest path problem using the Dijkstra algorithm. And ``pyepo.model.opt.optModel`` allows users to create a model in this way.

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
   model = myShortestPathModel(grid)
   cost = [random.random() for _ in range(model.num_cost)] # random cost vector
   model.setObj(cost) # set objective function
   sol, obj = model.solve() # solve
   # print res
   print('Obj: {}'.format(obj))
   for i, e in enumerate(model.arcs):
       if sol[i] > 1e-3:
           print(e)
