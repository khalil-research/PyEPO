Auto Grad Functions
+++++++++++++++++++

SPO+ Loss
=========

SPO+ Loss function [#f1]_, a surrogate loss function of SPO Loss, measures the decision error (optimality gap) of optimization problem.

.. autoclass:: pyepo.func.SPOPlus
    :members:

``pyepo.func.SPOPlus`` supports to solve optimization problems in parallel, parameter ``processes`` is the number of processors, **0 for using all available cores**.

.. code-block:: python

   import pyepo

   spo = pyepo.func.SPOPlus(optmodel, processes=2)


Diffenretiable Black-box Optimizer
==================================

Diffenretiable black-box (DBB) optimizer function [#f2]_ introduces optimizer block into neural networks.


.. autoclass:: pyepo.func.blackboxOpt
   :members:

``pyepo.func.blackboxOpt`` supports to solve optimization problems in parallel, parameter ``processes`` is the number of processors, 0 for using all available cores. ``lambd`` is a hyperparameter for function smoothing. The range of ``lambd`` should be **10** to **20**.

.. code-block:: python

   import pyepo

   dbb = pyepo.func.blackboxOpt(optmodel, lambd=10, processes=2)


Parallel Computation
====================

``PyEPO`` supports parallel computation for solving optimization problems in training, where the parameter ``processes`` is the number of processors to be used.

.. warning::  On Windows system, there is missing ``freeze_support`` to run ``multiprocessing`` directly from ``__main__``. When ``processes`` is not 1, try ``if __name__ == "__main__":`` instead of Jupyter notebook or a PY file.

.. rubric:: Footnotes

.. [#f1] Elmachtoub, A. N., & Grigas, P. (2021). Smart “predict, then optimize”. Management Science.
.. [#f2] Vlastelica, M., Paulus, A., Musil, V., Martius, G., & Rolínek, M. (2019). Differentiation of blackbox combinatorial solvers. arXiv preprint arXiv:1912.02175.
