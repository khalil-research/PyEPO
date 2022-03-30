Auto Grad Functions
+++++++++++++++++++

SPO+ Loss
=========

SPO+ Loss function, a surrogate loss function of SPO Loss, measures the decision error (optimality gap) of optimization problem.

.. autoclass:: pyepo.func.SPOPlus
    :members:

``pyepo.func.SPOPlus`` supports to solve optimization problems in parallel, parameter ``processes`` is the number of processors, **0 for using all available cores**.

.. code-block:: python

   import pyepo

   spo = pyepo.func.SPOPlus(optmodel, processes=2)


Diffenretiable Black-box Optimizer
==================================

Diffenretiable black-box (DBB) optimizer function introduces optimizer block into neural networks.


.. autoclass:: pyepo.func.blackboxOpt
   :members:

``pyepo.func.blackboxOpt`` supports to solve optimization problems in parallel, parameter ``processes`` is the number of processors, 0 for using all available cores. ``lambd`` is a hyperparameter for function smoothing. The range of ``lambd`` should be **10** to **20**.

.. code-block:: python

   import pyepo

   dbb = pyepo.func.blackboxOpt(optmodel, lambd=10, processes=2)
