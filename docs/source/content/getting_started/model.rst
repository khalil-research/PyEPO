Optimization Models
+++++++++++++++++++

``PyEPO`` trains predict-then-optimize models with a linear objective and unknown cost coefficients: only the cost is predicted, while the constraints are fixed.

``optModel`` is the interface that ``PyEPO`` trains against. It wraps an optimization solver or algorithm behind a ``setObj`` / ``solve`` contract. There are two ways to create one. For linear and integer programs supported by the DSL, define the problem with ``pyepo.dsl`` and compile it to a backend (:doc:`model/dsl`). For a custom algorithm or constraint generation, write an ``optModel`` subclass directly (:doc:`model/custom`).

``PyEPO`` also ships built-in models for shortest path, knapsack, traveling salesperson, capacitated vehicle routing, and portfolio problems, ready to pair with generated data (:doc:`model/builtin`).

For a runnable walkthrough, see the `01 Optimization Model <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/01%20Optimization%20Model.ipynb>`_ notebook.

.. toctree::
   :maxdepth: 1

   model/dsl
   model/custom
   model/builtin
