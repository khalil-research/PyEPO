Model
+++++

``PyEPO`` trains predict-then-optimize models with a linear objective and unknown cost coefficients: only the cost is predicted, while the constraints are fixed.

``optModel`` is the interface that ``PyEPO`` trains against. It wraps an optimization solver or algorithm behind a unified ``setObj`` / ``solve`` contract. There are two ways to produce one. Most users define the problem with the symbolic ``pyepo.dsl`` frontend and compile it to a backend. For full control, such as a custom algorithm or constraint generation, write an ``optModel`` subclass directly. Both are covered below.

When building a model, you do **not** specify the cost coefficients; they are predicted from data at training time.

``PyEPO`` also ships ready-made problems (shortest path, knapsack, traveling salesperson, capacitated vehicle routing, portfolio) for quick experiments. See the Data section for running them with generated data.

For a runnable walkthrough, see the `01 Optimization Model <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/01%20Optimization%20Model.ipynb>`_ notebook.


Defining Models with the DSL
============================

For most problems, define the model once with the symbolic ``pyepo.dsl`` frontend and compile it to a backend. Describe the variables with ``Variable``, the predicted cost with ``Parameter``, and the constraints, then call ``compile``. A multi-dimensional knapsack:

.. code-block:: python

   import numpy as np
   from pyepo import EPO, dsl

   weights = np.array([[3, 4, 3, 6, 4],
                       [4, 5, 2, 3, 5],
                       [5, 4, 6, 2, 3]])
   capacities = np.array([12, 10, 15])

   x = dsl.Variable(5, vtype=EPO.BINARY)          # decision variables
   c = dsl.Parameter(5)                           # the predicted cost
   optmodel = dsl.Problem(dsl.Maximize(c @ x),
                          [weights @ x <= capacities]).compile(backend="gurobi")

The compiled model is a standard ``optModel``. Set ``backend`` to ``"copt"``, ``"pyomo"``, ``"ortools"``, or ``"mpax"`` to use another solver; the generic backends take a ``solver=`` argument naming the open solver to run.

Variables take multi-dimensional shapes with numpy-style indexing and reductions, and constraints can be linear or quadratic. The objective can also mix a predicted cost with a known one, for example ``dsl.Minimize(c @ x + d @ y)`` predicts the cost on ``x`` and keeps ``d`` fixed on ``y``.

For a linear or quadratic objective with fixed constraints, the DSL is all you need. The rest of this page covers the lower-level ``optModel`` interface, for cases the DSL cannot express.


The optModel Interface
======================

The DSL compiles to an ``optModel``. Implement one directly when you need:

* a **custom solving algorithm**, for example a hand-written ADMM, a graph algorithm, or a heuristic, rather than a general solver;
* **constraint generation** through solver callbacks (lazy constraints, cutting planes), which a one-shot model definition cannot express;
* a structure the DSL does not model, such as a cost that broadcasts across several variables.

A subclass implements four methods:

* ``_getModel(self)``: build the model and return ``(model, variables)``. ``model`` is whatever ``solve`` needs (a solver model, a graph, or ``None``); ``variables`` sets ``num_cost``.
* ``setObj(self, c)``: store the cost vector ``c`` of length ``num_cost``.
* ``solve(self)``: solve and return ``(sol, obj)``. ``sol`` is a length-``num_cost`` array **aligned to the cost order** (``sol[i]`` is the value of the variable whose cost is ``c[i]``), and ``obj`` is the objective value.
* ``num_cost``: number of cost coefficients; defaults to ``len(self.x)``.

For a maximization problem, set ``self.modelSense = EPO.MAXIMIZE`` in ``__init__`` or ``_getModel``; the default is minimization.

.. autoclass:: pyepo.model.opt.optModel
    :noindex:
    :members: __init__, _getModel, setObj, solve, num_cost


Custom Algorithm
----------------

When the problem is solved by your own algorithm rather than a general solver, inherit from ``optModel`` and implement ``solve`` directly. Anything that returns a cost-aligned solution works: a graph algorithm, dynamic programming, a hand-written ADMM, or a heuristic. The example solves a grid shortest path with NetworkX and Dijkstra:

.. code-block:: python

   import numpy as np
   import networkx as nx

   from pyepo.model.opt import optModel

   class myShortestPathModel(optModel):

       def __init__(self, grid):
           self.grid = grid
           self.arcs = self._getArcs()      # list the grid edges
           super().__init__()

       def _getModel(self):
           g = nx.Graph()
           g.add_edges_from(self.arcs, cost=0)
           return g, g.edges                # variables set num_cost

       def setObj(self, c):
           for i, e in enumerate(self.arcs):
               self._model.edges[e]["cost"] = c[i]

       def solve(self):
           target = self.grid[0] * self.grid[1] - 1
           path = nx.shortest_path(self._model, weight="cost", source=0, target=target)
           active = set(zip(path[:-1], path[1:]))
           sol = np.zeros(self.num_cost)
           obj = 0.0
           for i, e in enumerate(self.arcs):
               if e in active:
                   sol[i] = 1                # cost-aligned solution
                   obj += self._model.edges[e]["cost"]
           return sol, obj

The same pattern wraps any algorithm: build state in ``_getModel``, store the cost in ``setObj``, and return a cost-aligned ``(sol, obj)`` from ``solve``.


Solver Backend Subclass
-----------------------

To use a solver's modeling API directly, inherit from a backend base class and implement ``_getModel``; ``setObj``, ``solve``, and ``num_cost`` come from the base. The base classes are ``optGrbModel`` (GurobiPy), ``optCoptModel`` (COPT), ``optOmoModel`` (Pyomo), ``optOrtModel`` or ``optOrtCpModel`` (OR-Tools), and ``optMpaxModel`` (MPAX). The GurobiPy version of the knapsack above:

.. code-block:: python

   import gurobipy as gp
   from gurobipy import GRB
   from pyepo.model.grb import optGrbModel

   class myKnapsack(optGrbModel):

       def _getModel(self):
           m = gp.Model()
           x = m.addVars(5, vtype=GRB.BINARY, name="x")
           m.modelSense = GRB.MAXIMIZE
           m.addConstr(3*x[0] + 4*x[1] + 3*x[2] + 6*x[3] + 4*x[4] <= 12)
           m.addConstr(4*x[0] + 5*x[1] + 2*x[2] + 3*x[3] + 5*x[4] <= 10)
           m.addConstr(5*x[0] + 4*x[1] + 6*x[2] + 2*x[3] + 3*x[4] <= 15)
           return m, x

COPT and Pyomo follow the same shape with their own APIs. Pyomo and OR-Tools take a ``solver=`` argument and require ``self.modelSense`` to be set in ``_getModel``; OR-Tools provides both ``optOrtModel`` (pywraplp, LP and MIP) and ``optOrtCpModel`` (CP-SAT). See the API reference for each base class.

MPAX has no solver model object, so ``_getModel`` fills the standard-form matrices and returns ``(None, [])``:

.. code-block:: python

   import jax.numpy as jnp
   from pyepo import EPO
   from pyepo.model.mpax import optMpaxModel

   class myMpaxKnapsack(optMpaxModel):
       use_sparse_matrix = False

       def _getModel(self):
           n = 5
           self.A = jnp.zeros((0, n))                       # equality A x = b (none)
           self.b = jnp.zeros((0,))
           self.G = -jnp.array(weights, dtype=jnp.float32)  # inequality G x >= h
           self.h = -jnp.array(capacities, dtype=jnp.float32)
           self.l = jnp.zeros(n)                            # bounds x in [0, 1]
           self.u = jnp.ones(n)
           self.modelSense = EPO.MAXIMIZE
           return None, []

Assign ``self.Q`` a PSD matrix for a convex QP objective. See ``optMpaxModel`` for the full matrix contract.


Constraint Generation
---------------------

Some problems have too many constraints to write up front. The traveling salesperson problem, for instance, has exponentially many subtour-elimination constraints. The standard approach is constraint generation: solve a relaxed model, inspect the solution for violations, add the violated constraints, and re-solve until none remain. With a solver this runs through a lazy-constraint callback, where the solver calls back into your code whenever it finds an integer solution and you add the violated constraints on the fly.

The DSL is a one-shot model definition and cannot express this, so such problems are written as a backend subclass that registers the callback in ``_getModel``. The shape is:

.. code-block:: python

   import gurobipy as gp
   from pyepo.model.grb import optGrbModel

   class myTSP(optGrbModel):

       def _getModel(self):
           m = gp.Model()
           # ... edge variables x and degree constraints ...
           m.Params.lazyConstraints = 1
           return m, x

       def solve(self):
           self._model.optimize(self._subtourCallback)   # add subtours lazily
           # ... read the tour from the active edges ...

``pyepo.model.grb.tspDFJModel`` is a complete implementation for the TSP.
