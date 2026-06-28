Solver Backends
+++++++++++++++

``PyEPO`` separates the training frontend from the optimization backend. A backend supplies the ``optModel`` interface: update the objective with ``setObj`` and solve the resulting optimization problem with ``solve``.

Supported backends:

* ``gurobi``: GurobiPy backend.
* ``copt``: COPT backend.
* ``pyomo``: Pyomo backend for open or commercial solvers.
* ``ortools``: Google OR-Tools backend.
* ``mpax``: JAX/MPAX backend for LP/QP solving on GPU.

For model definition, backend selection, and solver parameters, see :doc:`getting_started/model`.

For installation notes, including solver packages and license-free options, see :doc:`install`.
