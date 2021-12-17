Model
+++++

``spo`` contains several pre-defined optimizations models with GurobiPy and Pyomo. It includes the shortest path problem (GurobiPy & Pyomo), the knapsack problem (GurobiPy & Pyomo), and the traveling salesman problem (GurobiPy).

Our API is also designed to support users to define their own problems based on GurobiPy and Pyomo. Besides the API of GurobiPy & Pyomo, users can also build problems from scratch with whatever solvers and algorithms they want to use.

To build optimizations models with ``spo``, users do **not** need specific costs and objective functions since the cost vector is unknown but can be estimated from data.

.. warning:: For convenience, optimization problems in ``spo`` always **minimize** the cost. Therefore, for maximization problems, we need convert them into minimization by multiplying the cost vector with -1.

Optimizations model in ``spo`` is an object of ``optModel``. The following code snippets use ``spo.model`` to build ``optModel``:


Pre-defined Models
==================

Pre-defined models are some classic optimization problems, including shortest path, knapsack, and traveling salesman.


Shortest Path
-------------

It is a (h,w) grid network and the goal is to find the shortest path from northwest to southeast. In our examples, the grid size of network is (5,5).

.. image:: ../../images/shortestpath.png
  :width: 300
  :alt: Shortest Path on the Grid Graph

The shortest path problem is built as Linear programming (LP) and formulated as a minimum cost flow problem. Thus, network flow constraints are modeled as the feasible region.

Shortest Path GurobiPy Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``optModel`` is built from ``spo.model.grb.shortestPathModel``, in which API uses GurobiPy to model the shortest path problem.

.. autoclass:: spo.model.grb.shortestPathModel
   :members: __init__, setObj, solve, num_cost

.. code-block:: python

   import spo

   grid = (5,5) # network grid
   sp_model = spo.model.grb.shortestPathModel(grid) # build model

Users can use ``setObj`` with a specific cost vector to set current objective function and use ``solve`` to solve it.

.. code-block:: python

   import random
   cost = [random.random() for _ in range(sp_model.num_cost)] # random cost vector
   sp_model.setObj(cost) # set objective function
   sp_model.solve() # solve

Shortest Path Pyomo Model
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``optModel`` is built from ``spo.model.omo.shortestPathModel``, in which API uses Pyomo to model the shortest path problem.

.. autoclass:: spo.model.omo.shortestPathModel
   :members: __init__, setObj, solve, num_cost

Pyomo supports a wide variety of solvers in the background (e.g. BARON, CBC, CPLEX, and Gurobi). ``spo.model.omo.shortestPathModel`` support users to call different solvers with class parameter ``solver``.

.. code-block:: python

   import spo

   grid = (5,5) # network grid
   sp_model = spo.model.omo.shortestPathModel(grid, solver="glpk") # build model with glpk
   sp_model = spo.model.omo.shortestPathModel(grid, solver="gurobi") # build model with gurobi

You can get the current list of supported solvers using the pyomo command:

.. code-block:: bash

   pyomo help --solvers

Same as ``spo.model.grb.shortestPathModel``, methods ``setObj`` and ``solve`` can set objective function and solve the problem.

.. code-block:: python

   import random
   cost = [random.random() for _ in range(sp_model.num_cost)] # random cost vector
   sp_model.setObj(cost) # set objective function
   sp_model.solve() # solve


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

.. note:: The dimension of the knapsack and the number of items are implicitly defined by the array of **weights** and **capacities**.

Knapsack GurobiPy Model
^^^^^^^^^^^^^^^^^^^^^^^

The ``optModel`` is built from ``spo.model.grb.knapsackModel``, in which API uses GurobiPy to model the knapsack problem.

.. autoclass:: spo.model.grb.knapsackModel
   :members: __init__, setObj, solve, num_cost, relax

.. code-block:: python

   import spo

   weights = [[3, 4, 3, 6, 4],
              [4, 5, 2, 3, 5],
              [5, 4, 6, 2, 3]] # constraints coefficients
   capacities = [12, 10, 15] # constraints rhs
   ks_model = spo.model.grb.knapsackModel(weights, capacities) # build model

Users can use ``setObj`` with a specific cost vector to set current objective function and use ``solve`` to solve it.

.. warning:: Since knapsack is a maximization problem, the cost vector should multiply with -1.

.. code-block:: python

   import random
   cost = [- random.random() for _ in range(ks_model.num_cost)] # random cost vector multiply with -1
   ks_model.setObj(cost) # set objective function
   ks_model.solve() # solve

In mathematics, the relaxation of a (Mixed) Integer Linear Programming is the problem that arises by removing the integrality constraint of each variable. As an ILP, ``ks_model`` allows users to relax ILP with ``relax`` method to obtain a new relaxed ``optModel``.

.. code-block:: python

   ks_model_rel = ks_model.relax() # relax

Knapsack Pyomo Model
^^^^^^^^^^^^^^^^^^^^

The ``optModel`` is built from ``spo.model.omo.knapsackModel``, in which API uses Pyomo to model the knapsack problem.

.. autoclass:: spo.model.omo.knapsackModel
   :members: __init__, setObj, solve, num_cost, relax

.. code-block:: python

   import spo

   weights = [[3, 4, 3, 6, 4],
              [4, 5, 2, 3, 5],
              [5, 4, 6, 2, 3]] # constraints coefficients
   capacities = [12, 10, 15] # constraints rhs
   ks_model = spo.model.omo.knapsackModel(weights, capacities, solver="glpk") # build model with glpk
   ks_model = spo.model.omo.knapsackModel(weights, capacities, solver="gurobi") # build model with gurobi

Same as ``spo.model.grb.knapsackModel``,  users can use ``setObj``, ``solve``, and ``relax`` methods.

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

.. autoclass:: spo.model.grb.tspDFJModel
   :members: __init__, setObj, solve, num_cost

The number of subtour elimination constraints for DFJ formulation is exponential. Thus, we solved it with column generation. Because of that, the linear relaxation of DFJ is **not** supported in our implementation.

Same as previous model, the code for traveling salesman problem with DFJ formulation is as follows:

.. code-block:: python

   import spo
   import random

   num_nodes = 20 # number of nodes
   tsp_model = spo.model.grb.tspDFJModel(num_nodes) # build model

   cost = [random.random() for _ in range(tsp_model.num_cost)] # random cost vector
   tsp_model.setObj(cost) # set objective function
   tsp_model.solve() # solve


GG formulation
^^^^^^^^^^^^^^

.. autoclass:: spo.model.grb.tspGGModel
   :members: __init__, setObj, solve, num_cost, relax

Same as previous model, the code for traveling salesman problem with GG formulation is as follows:

.. code-block:: python

   import spo
   import random

   num_nodes = 20 # number of nodes
   tsp_model = spo.model.grb.tspGGModel(num_nodes) # build model

   cost = [random.random() for _ in range(tsp_model.num_cost)] # random cost vector
   tsp_model.setObj(cost) # set objective function
   tsp_model.solve() # solve

   tsp_model.relax() # relax


MTZ formulation
^^^^^^^^^^^^^^^

.. autoclass:: spo.model.grb.tspMTZModel
   :members: __init__, setObj, solve, num_cost, relax

Same as previous model, the code for traveling salesman problem with MTZ formulation is as follows:

.. code-block:: python

   import spo
   import random

   num_nodes = 20 # number of nodes
   tsp_model = spo.model.grb.tspMTZModel(num_nodes) # build model

   cost = [random.random() for _ in range(tsp_model.num_cost)] # random cost vector
   tsp_model.setObj(cost) # set objective function
   tsp_model.solve() # solve

   tsp_model.relax() # relax


User-defined Models
===================

User can build optimization problem with linear objective function.


GurobiPy Models
---------------

User-defined models with GurobiPy can be easily defined by the inheritance of the abstract class ``spo.model.grb.optGRBModel``.

.. autoclass:: spo.model.grb.optGRBModel
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

In the general case, users only need to implement ``_getModel`` and  ``num_cost`` method with GurobiPy.

.. code-block:: python

   import gurobipy as gp
   from gurobipy import GRB

   from spo.model.grb import optGRBModel

   class myModel(optGRBModel):

       def _getModel(self):
           # ceate a model
           m = gp.Model()
           # varibles
           x = m.addVars(5, name="x", vtype=GRB.BINARY)
           # sense (must be minimize)
           m.modelSense = GRB.MINIMIZE
           # constraints
           m.addConstr(3 * x[0] + 4 * x[1] + 3 * x[2] + 6 * x[3] + 4 * x[4] <= 12)
           m.addConstr(4 * x[0] + 5 * x[1] + 2 * x[2] + 3 * x[3] + 5 * x[4] <= 10)
           m.addConstr(5 * x[0] + 4 * x[1] + 6 * x[2] + 2 * x[3] + 3 * x[4] <= 15)
           return m, x

       @property
       def num_cost(self):
           return 5

   model = myModel()
   cost = [- random.random() for _ in range(model.num_cost)] # random cost vector
   model.setObj(cost) # set objective function
   model.solve() # solve


Pyomo Models
------------

User-defined models with Pyomo can be easily defined by the inheritance of the abstract class ``spo.model.omo.optOmoModel``.

.. autoclass:: spo.model.omo.optOmoModel
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

In the general case, users only need to implement ``_getModel`` and  ``num_cost`` method with Pyomo.

.. code-block:: python

   from pyomo import environ as pe

   from spo.model.omo import optOmoModel

   class myModel(optOmoModel):

       def _getModel(self):
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

       @property
       def num_cost(self):
           return 5

   model = myModel()
   cost = [- random.random() for _ in range(model.num_cost)] # random cost vector
   model.setObj(cost) # set objective function
   model.solve() # solve
