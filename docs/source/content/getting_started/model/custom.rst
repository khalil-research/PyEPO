Custom Models
+++++++++++++

The DSL compiles to an ``optModel``. Implement an ``optModel`` directly when you need:

* a **custom solving algorithm**, such as a hand-written ADMM method, a graph algorithm, or a heuristic, rather than a general solver;
* **constraint generation** through solver callbacks (lazy constraints, cutting planes), which a one-shot model definition cannot express.

A subclass implements the solving interface:

* ``_getModel(self)``: build the model and return ``(model, variables)``. ``model`` is whatever ``solve`` needs (a solver model, a graph, or ``None``). ``variables`` sets ``num_cost``.
* ``setObj(self, c)``: store the cost vector ``c`` of length ``num_cost``.
* ``solve(self)``: solve and return ``(sol, obj)``. ``sol`` is a length-``num_cost`` array **aligned to the cost order** (``sol[i]`` is the value of the variable whose cost is ``c[i]``), and ``obj`` is the objective value.
* ``num_cost``: number of cost coefficients. Defaults to ``len(self.x)``.

Constructor arguments are captured automatically for ``rebuild()`` and multiprocessing, so most subclasses only need to define the solving interface. If a model constructor needs special handling, override ``get_config``.

For a maximization problem, set ``self.modelSense = EPO.MAXIMIZE`` in ``__init__`` or ``_getModel``. The default is minimization. The Gurobi and COPT bases detect the sense from the solver model automatically.

.. autoclass:: pyepo.model.opt.optModel
    :noindex:
    :members: __init__, _getModel, setObj, solve, num_cost, rebuild


Custom Algorithm
================

When the problem is solved by your own algorithm rather than a general solver, inherit from ``optModel`` and implement ``solve`` directly. Anything that returns a cost-aligned solution can be wrapped this way: a graph algorithm, dynamic programming, a hand-written ADMM method, or a heuristic. The example below solves a grid shortest path with NetworkX and Dijkstra:

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
=======================

To use a solver's modeling API directly, inherit from a backend base class and implement ``_getModel``. ``setObj``, ``solve``, and ``num_cost`` come from the base class. The GurobiPy version of a binary program with linear constraints:

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

The other backends follow the same shape with their own APIs: ``optCoptModel`` (COPT), ``optOmoModel`` (Pyomo), and ``optOrtModel`` / ``optOrtCpModel`` (OR-Tools). Of these, ``optOmoModel`` and ``optOrtModel`` take a ``solver=`` argument. ``optOrtCpModel`` (CP-SAT) is integer-only with a fixed solver. ``optMpaxModel`` is different: it has no solver model object, so ``_getModel`` fills the standard-form matrices ``A``, ``b``, ``G``, ``h``, ``l``, ``u`` (and an optional positive semidefinite ``Q``) and returns ``(None, [])``.

.. autoclass:: pyepo.model.mpax.optMpaxModel
    :noindex:
    :members: __init__, _getModel, setObj, solve, num_cost


Constraint Generation
=====================

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

``pyepo.model.grb.tspDFJModel`` implements this pattern for the TSP.
