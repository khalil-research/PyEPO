DSL Models
++++++++++

Describe the problem once with ``Variable``, ``Parameter``, and constraints, then compile it to a backend. The example below is a binary program with a predicted cost and linear constraints:

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

All backends share this interface and are selected with ``backend=``. Gurobi and COPT are commercial solvers. Pyomo and OR-Tools can use open solvers such as HiGHS, GLPK, CBC, and SCIP. MPAX solves linear and quadratic programs on the GPU.

``compile`` forwards keyword arguments to the backend. ``solver=`` applies only to the generic backends (``pyomo`` / ``ortools``) and names the solver they run. ``timelimit=`` sets a time limit in seconds where the backend supports one. Any other keyword is passed through as a native solver parameter where the backend accepts one:

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
     - maps to the chosen solver's own option for GLPK, CBC, SCIP, HiGHS, Ipopt, Gurobi, or CPLEX. With any other solver, pass the native option as a keyword instead
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
=========

A ``Variable`` takes a shape (an integer or a tuple), an optional ``vtype``, and bounds. A problem declares exactly one ``Parameter``, the predicted cost.

.. code-block:: python

   x = dsl.Variable(5)                            # continuous (the default)
   x = dsl.Variable(5, vtype=EPO.BINARY)          # also INTEGER, CONTINUOUS, or a per-entry list
   x = dsl.Variable((3, 3))                       # multi-dimensional (a tuple shape)
   x = dsl.Variable(5, lb=0, ub=1)                # bounds, scalar or array

   c = dsl.Parameter(5)                           # the predicted cost


Objectives
==========

Whether a coefficient is predicted or known is determined by its type: a ``Parameter`` is predicted (``c``), while a numpy array is fixed (``d``, ``Q``). The predicted cost enters linearly. A known quadratic term may also be added. Below, ``d`` is a numpy array, ``y`` is another ``Variable``, ``Q`` is a numpy matrix, and ``k`` is an index.

.. code-block:: python

   dsl.Minimize(c @ x)                            # inner product (scalar); or dsl.Maximize(c @ x)
   dsl.Minimize((c * x).sum())                    # elementwise then reduce; same objective as c @ x
   dsl.Minimize(c @ x + d @ y)                    # predict c on x, keep known d on y
   dsl.Minimize(c @ x[:k] + d @ x[k:])            # predict part of one variable, fix the rest
   dsl.Minimize((d + c) @ x)                      # a known base d plus the predicted c
   dsl.Minimize(c @ x + x @ Q @ x)                # predicted linear plus a known quadratic

``c @ x`` is a 1-D inner product. For a multi-dimensional cost, use ``(c * x).sum()`` (elementwise, then reduced). A quadratic objective term needs a backend with QP support: Gurobi, COPT, MPAX, or Pyomo with a QP-capable solver.

.. note:: A quadratic objective term is solve-only. Compiling warns, and ``pyepo.func`` training methods and ``pyepo.metric`` metrics reject the model with an error. Quadratic constraints carry no such restriction.


Constraints
===========

Constraints are fixed across instances. Only the cost is predicted. Pass constraints as a list to ``Problem``.

.. code-block:: python

   A @ x <= b                                     # linear: <=, >=, ==
   x.sum() == 1                                   # reduction
   x.sum(axis=1) == 1                             # per-axis sums, e.g. an assignment
   x @ Q @ x <= gamma                             # quadratic (Gurobi, COPT, or QP-capable Pyomo)

For a linear or quadratic objective with fixed constraints, the DSL is all you need. For cases the DSL cannot express, see :doc:`custom`.
