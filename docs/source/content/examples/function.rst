Auto Grad Functions
+++++++++++++++++++

Smart Predict-then-Optimize Loss+ (SPO+)
========================================

SPO+ Loss function [#f1]_, a surrogate loss function of SPO Loss, measures the decision error (optimality gap) of optimization problem.

.. autoclass:: pyepo.func.SPOPlus
    :members:

``pyepo.func.SPOPlus`` supports to solve optimization problems in parallel, parameter ``processes`` is the number of processors, **0 for using all available cores**.

.. code-block:: python

   import pyepo

   spo = pyepo.func.SPOPlus(optmodel, processes=2)


Diffenretiable Black-box Optimizer (DBB)
========================================

Diffenretiable black-box (DBB) optimizer function [#f2]_ introduces optimizer block into neural networks.


.. autoclass:: pyepo.func.blackboxOpt
   :members:

``pyepo.func.blackboxOpt`` supports to solve optimization problems in parallel, parameter ``processes`` is the number of processors, 0 for using all available cores. ``lambd`` is a hyperparameter for function smoothing. The range of ``lambd`` should be **10** to **20**.

.. code-block:: python

   import pyepo

   dbb = pyepo.func.blackboxOpt(optmodel, lambd=10, processes=2)


Perturbed Optimizer (PO)
========================

Perturbed Optimizer (PO) [#f3]_ uses Monte-Carlo samples to estimate solutions, which makes it differentiable.


.. autoclass:: pyepo.func.perturbedOpt
   :members:

``pyepo.func.perturbedOpt`` supports to solve optimization problems in parallel, parameter ``processes`` is the number of processors, 0 for using all available cores. ``n_samples`` is the number of Monte-Carlo samples to estimate solutions, and ``epsilon`` is the variance of Gaussian noise perturbation.

.. code-block:: python

   import pyepo

   dbb = pyepo.func.perturbedOpt(optmodel, n_samples=10, epsilon=0.5, processes=2)


Perturbed Fenchel-Young loss (PYFL)
===================================

Perturbed Fenchel-Young loss (PYFL) function [#f3]_ uses Monte-Carlo samples to estimate solutions as PO, and the Fenchel-Young loss allows to directly optimize a loss between the features and solutions with less computation.


.. autoclass:: pyepo.func.perturbedFenchelYoung
   :members:

``pyepo.func.perturbedFenchelYoung`` supports to solve optimization problems in parallel, parameter ``processes`` is the number of processors, 0 for using all available cores. ``n_samples`` is the number of Monte-Carlo samples to estimate solutions, and ``epsilon`` is the variance of Gaussian noise perturbation.

.. code-block:: python

   import pyepo

   dbb = pyepo.func.perturbedFenchelYoung(optmodel, n_samples=10, epsilon=0.5, processes=2)


Parallel Computation
====================

``PyEPO`` supports parallel computation for solving optimization problems in training, where the parameter ``processes`` is the number of processors to be used.

.. warning::  On Windows system, there is missing ``freeze_support`` to run ``multiprocessing`` directly from ``__main__``. When ``processes`` is not 1, try ``if __name__ == "__main__":`` instead of Jupyter notebook or a PY file.

.. rubric:: Footnotes

.. [#f1] Elmachtoub, A. N., & Grigas, P. (2021). Smart “predict, then optimize”. Management Science.
.. [#f2] Vlastelica, M., Paulus, A., Musil, V., Martius, G., & Rolínek, M. (2019). Differentiation of blackbox combinatorial solvers. arXiv preprint arXiv:1912.02175.
.. [#f3] Berthet, Q., Blondel, M., Teboul, O., Cuturi, M., Vert, J. P., & Bach, F. (2020). Learning with differentiable perturbed optimizers. Advances in neural information processing systems, 33, 9508-9519.
