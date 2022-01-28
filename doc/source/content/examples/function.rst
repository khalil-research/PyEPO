Functions
+++++++++

SPO+ Loss
=========

SPO+ Loss function, a surrogate loss function of SPO Loss, which measures the decision error (optimality gap) of optimization problem.

For SPO/SPO+ Loss, the objective function is linear and constraints are known and fixed, but the cost vector need to be predicted from contextual data.

The SPO+ Loss is convex with subgradient. Thus, allows us to design an algorithm based on stochastic gradient descent.

.. autoclass:: pyepo.func.SPOPlus
    :members:

``pyepo.func.SPOPlus`` supports to solve optimization problems in parallel, parameter ``processes`` is the number of processors, **0 for using all available cores**.

.. code-block:: python

   import pyepo

   spo = pyepo.func.SPOPlus(sp_model, processes=8)


Diffenretiable Black-box
========================

Diffenretiable block-box optimizer function, which introduce blocks into neural networks.

For DBB, the objective function is linear and constraints are known and fixed, but the cost vector need to be predicted from contextual data.

The block-box approximate gradient of optimizer smoothly. Thus, allows us to design an algorithm based on stochastic gradient descent.

.. autoclass:: pyepo.func.blackboxOpt
   :members:

``pyepo.func.blackboxOpt`` supports to solve optimization problems in parallel, parameter ``processes`` is the number of processors, 0 for using all available cores. ``lambd`` is a hyperparameter for function smoothing. The range of ``lambd`` should be **10** to **20**.

.. code-block:: python

   import pyepo

   dbb = pyepo.func.blackboxOpt(sp_model, lambd=10, processes=8)
