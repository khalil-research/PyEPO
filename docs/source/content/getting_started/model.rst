Optimization Models
+++++++++++++++++++

``PyEPO`` trains predict-then-optimize models with a linear objective and unknown cost coefficients: only the cost is predicted, while the constraints are fixed.

``optModel`` is the interface that ``PyEPO`` trains against. It wraps an optimization solver or algorithm behind a ``setObj`` / ``solve`` contract. There are two ways to produce one. For linear and integer programs supported by the DSL, define the problem with ``pyepo.dsl`` and compile it to a backend. For a custom algorithm or constraint generation, write an ``optModel`` subclass directly. Both are covered below.

When building a model, you do **not** specify the cost coefficients; they are predicted from data at training time.

``PyEPO`` also includes built-in problem models: shortest path, knapsack, traveling salesperson, capacitated vehicle routing, and portfolio. See :doc:`data` for running them with generated data.

For a runnable walkthrough, see the `01 Optimization Model <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/01%20Optimization%20Model.ipynb>`_ notebook.


Defining Models with the DSL
============================

Describe the problem once with ``Variable``, ``Parameter``, and constraints, then compile it to a backend. A binary program with a predicted cost and linear constraints:

.. code-block:: python

   import numpy as np
   from pyepo import EPO, dsl

   A = np.array([[3, 4, 3, 6, 4],
                 [4, 5, 2, 3, 5],
                 [5, 4, 6, 2, 3]])
   b = np.array([12, 10, 15])

   x = dsl.Variable(5, vtype=EPO.BINARY)          # decision variables
   c = dsl.Parameter(5)                           # the predicted cost
   prob = dsl.Problem(dsl.Maximize(c @ x), [A @ x <= b])
   optmodel = prob.compile(backend="gurobi")      # compile to a solver backend

The compiled model is an ``optModel``. During training, ``pyepo.func`` calls ``setObj`` and ``solve``.

All backends share this interface and are selected with ``backend=``. Gurobi and COPT are commercial solvers. Pyomo and OR-Tools can use open solvers such as HiGHS, GLPK, CBC, and SCIP. MPAX solves linear and quadratic programs on GPU. The generic backends take a ``solver=`` argument naming the solver to run.

``compile`` forwards keyword arguments to the backend. ``solver=`` applies only to the generic backends (``pyomo`` / ``ortools``) and names the solver they run; ``timelimit=`` (seconds) sets a time limit where the backend supports one; any other keyword is passed to the solver as a native parameter:

.. list-table::
   :header-rows: 1
   :widths: 12 26 34 28

   * - Backend
     - ``solver=``
     - ``timelimit=`` (seconds)
     - Other keywords
   * - ``gurobi``
     - not applicable
     - maps to ``TimeLimit``
     - any native Gurobi parameter, e.g. ``MIPGap=0.01``
   * - ``copt``
     - not applicable
     - maps to ``TimeLimit``
     - any native COPT parameter
   * - ``pyomo``
     - open solver name (default ``"glpk"``)
     - maps to the chosen solver's own option (known for GLPK, CBC, SCIP, HiGHS, Ipopt, Gurobi, CPLEX; for other solvers pass the native option)
     - passed through as solver options
   * - ``ortools``
     - pywraplp solver name (default ``"scip"``)
     - supported
     - not accepted
   * - ``mpax``
     - not applicable
     - accepted and ignored (MPAX exposes no time-limit setting)
     - not accepted

.. code-block:: python

   prob.compile(backend="pyomo", solver="appsi_highs")    # an open solver via Pyomo
   prob.compile(backend="gurobi", timelimit=10)           # time limit (seconds)
   prob.compile(backend="gurobi", MIPGap=0.01)            # native solver parameters pass through


Variables
---------

A ``Variable`` takes a shape (an integer or a tuple), an optional ``vtype``, and bounds. A problem declares exactly one ``Parameter``, the predicted cost.

.. code-block:: python

   x = dsl.Variable(5)                            # continuous (the default)
   x = dsl.Variable(5, vtype=EPO.BINARY)          # also INTEGER, CONTINUOUS, or a per-entry list
   x = dsl.Variable((3, 3))                       # multi-dimensional (a tuple shape)
   x = dsl.Variable(5, lb=0, ub=1)                # bounds, scalar or array

   c = dsl.Parameter(5)                           # the predicted cost


Objectives
----------

Whether a coefficient is predicted or known is decided by its type: a ``Parameter`` is predicted (``c``), while a numpy array is fixed (``d``, ``Q``). The predicted cost enters linearly; a known quadratic term may be added. Below, ``d`` is a numpy array, ``y`` another ``Variable``, ``Q`` a numpy matrix, and ``k`` an index.

.. code-block:: python

   dsl.Minimize(c @ x)                            # inner product (scalar); or dsl.Maximize(c @ x)
   dsl.Minimize((c * x).sum())                    # elementwise then reduce; same objective as c @ x
   dsl.Minimize(c @ x + d @ y)                    # predict c on x, keep known d on y
   dsl.Minimize(c @ x[:k] + d @ x[k:])            # predict part of one variable, fix the rest
   dsl.Minimize((d + c) @ x)                      # a known base d plus the predicted c
   dsl.Minimize(c @ x + x @ Q @ x)                # predicted linear plus a known quadratic

``c @ x`` is a 1-D inner product; for a multi-dimensional cost use ``(c * x).sum()`` (elementwise, then reduced). A quadratic objective term needs a backend with QP support (Gurobi, COPT, or MPAX).


Constraints
-----------

Constraints are fixed across instances; only the cost is predicted. Pass them as a list to ``Problem``.

.. code-block:: python

   A @ x <= b                                     # linear: <=, >=, ==
   x.sum() == 1                                   # reduction
   x.sum(axis=1) == 1                             # per-axis sums, e.g. an assignment
   x @ Q @ x <= gamma                             # quadratic (Gurobi and COPT)

For a linear or quadratic objective with fixed constraints, the DSL is all you need; the rest of this page is the lower-level ``optModel`` interface for cases it cannot express.


The optModel Interface
======================

The DSL compiles to an ``optModel``. Implement one directly when you need:

* a **custom solving algorithm**, for example a hand-written ADMM, a graph algorithm, or a heuristic, rather than a general solver;
* **constraint generation** through solver callbacks (lazy constraints, cutting planes), which a one-shot model definition cannot express.

A subclass implements the solving interface plus an explicit reconstruction
configuration:

* ``_getModel(self)``: build the model and return ``(model, variables)``. ``model`` is whatever ``solve`` needs (a solver model, a graph, or ``None``); ``variables`` sets ``num_cost``.
* ``setObj(self, c)``: store the cost vector ``c`` of length ``num_cost``.
* ``solve(self)``: solve and return ``(sol, obj)``. ``sol`` is a length-``num_cost`` array **aligned to the cost order** (``sol[i]`` is the value of the variable whose cost is ``c[i]``), and ``obj`` is the objective value.
* ``num_cost``: number of cost coefficients; defaults to ``len(self.x)``.
* ``get_config(self)``: return the constructor arguments needed to build a fresh equivalent model. This is used by multiprocessing, scorers, and ``rebuild()``; do not include solver state or the current objective.

For a maximization problem, set ``self.modelSense = EPO.MAXIMIZE`` in ``__init__`` or ``_getModel``; the default is minimization.

.. autoclass:: pyepo.model.opt.optModel
    :noindex:
    :members: __init__, _getModel, setObj, solve, num_cost, get_config, rebuild, to_spec


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

       def get_config(self):
           return {**super().get_config(), "grid": self.grid}

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

To use a solver's modeling API directly, inherit from a backend base class and implement ``_getModel``; ``setObj``, ``solve``, and ``num_cost`` come from the base. The GurobiPy version of the program above:

.. code-block:: python

   import gurobipy as gp
   from gurobipy import GRB
   from pyepo.model.grb import optGrbModel

   class myModel(optGrbModel):

       def _getModel(self):
           m = gp.Model()
           x = m.addVars(5, vtype=GRB.BINARY, name="x")
           m.modelSense = GRB.MAXIMIZE
           m.addConstr(3*x[0] + 4*x[1] + 3*x[2] + 6*x[3] + 4*x[4] <= 12)
           m.addConstr(4*x[0] + 5*x[1] + 2*x[2] + 3*x[3] + 5*x[4] <= 10)
           m.addConstr(5*x[0] + 4*x[1] + 6*x[2] + 2*x[3] + 3*x[4] <= 15)
           return m, x

The other backends follow the same shape with their own APIs: ``optCoptModel`` (COPT), ``optOmoModel`` (Pyomo), and ``optOrtModel`` / ``optOrtCpModel`` (OR-Tools), which take a ``solver=`` argument. ``optMpaxModel`` is different: it has no solver model object, so ``_getModel`` fills the standard-form matrices ``A``, ``b``, ``G``, ``h``, ``l``, ``u`` (and an optional PSD ``Q``) and returns ``(None, [])``.

.. autoclass:: pyepo.model.mpax.optMpaxModel
    :noindex:
    :members: __init__, _getModel, setObj, solve, num_cost


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
