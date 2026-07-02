Solver Backends
+++++++++++++++

``PyEPO`` separates the training frontend from the optimization backend. A
backend supplies the ``optModel`` interface: update the objective with
``setObj``, solve with ``solve``. The training methods depend only on this
interface, so the same training code runs on any backend.

Supported Backends
==================

.. list-table::
   :header-rows: 1
   :widths: 10 22 26 42

   * - Backend
     - License
     - Problem classes
     - Notes
   * - ``gurobi``
     - commercial, free academic license
     - LP, MIP, QP, quadratic constraints
     - the default backend; supports lazy-constraint callbacks (TSP DFJ, VRP RCI) and cut recycling
   * - ``copt``
     - commercial, free academic license
     - LP, MIP, QP, quadratic constraints
     - same callback support as Gurobi
   * - ``pyomo``
     - open (the modeling layer)
     - whatever the chosen solver supports
     - ``solver=`` names the engine: GLPK, CBC, HiGHS, SCIP, or a licensed solver; QP needs a QP-capable solver
   * - ``ortools``
     - open
     - LP, MIP (no quadratics)
     - pywraplp with ``solver=`` (default ``"scip"``); a CP-SAT variant handles integer-only models
   * - ``mpax``
     - open
     - LP, QP
     - JAX-based first-order (PDHG) solver on GPU; solves whole batches at once; **continuous only**: integer variables are relaxed with a warning

Selecting a Backend
===================

The DSL ``compile`` and the built-in model factories take a ``backend``
keyword, so the same problem definition runs on any installed backend:

.. code-block:: python

   optmodel = prob.compile(backend="gurobi")
   optmodel = prob.compile(backend="pyomo", solver="appsi_highs")   # open solver
   optmodel = prob.compile(backend="mpax")                          # LP/QP on GPU

   model = pyepo.model.shortestPathModel(grid)                      # default Gurobi
   model = pyepo.model.shortestPathModel(grid, backend="ortools", solver="scip")

Solver Parameters
=================

``compile`` and the model factories forward keyword arguments to the backend.
``timelimit=`` (seconds) sets a time limit where supported. Other keywords
pass through as native solver parameters where the backend accepts them.

.. code-block:: python

   prob.compile(backend="gurobi", timelimit=10)
   prob.compile(backend="gurobi", MIPGap=0.01)

The full per-backend keyword table is in :doc:`getting_started/model`.

MPAX: GPU Batch Solving
=======================

MPAX solves an entire mini-batch in one GPU dispatch, so ``optDataset``
construction and every training step avoid the per-instance solver loop. With
the JAX frontend the solve is traceable, and the whole training step compiles
under ``jax.jit`` (see :doc:`frontends/jax`). The PyTorch frontend uses the
same batched solve through a dlpack bridge. PDHG is a continuous first-order
method, so integer and binary variables are relaxed to their bounds and
solutions may be fractional. PyEPO warns when this happens.

Related Pages
=============

* :doc:`getting_started/model` documents model definition, backend selection, and the solver-parameter table.
* :doc:`install` lists the pip extras that install each backend.
