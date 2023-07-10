Auto Grad Functions
+++++++++++++++++++

Smart Predict-then-Optimize Loss+ (SPO+)
========================================

SPO+ Loss function [#f1]_ is a surrogate loss function of SPO Loss (Regret), which measures the decision error of optimization problem. For SPO/SPO+ Loss, the objective function is linear and constraints are known and fixed, but the cost vector need to be predicted from contextual data. The SPO+ Loss is convex with non-zero subgradient. Thus, allows us to design an algorithm based on stochastic gradient descent.

.. autoclass:: pyepo.func.SPOPlus
    :noindex:
    :members:

``pyepo.func.SPOPlus`` supports to solve optimization problems in parallel, parameter ``processes`` is the number of processors, **0 for using all available cores**.

.. code-block:: python

   import pyepo

   spo = pyepo.func.SPOPlus(optmodel, processes=2)


Differentiable Black-box Optimizer (DBB)
========================================

Diffenretiable black-box (DBB) optimizer function [#f2]_ estimates gradients from interpolation, replacing the zero gradients. For differentiable block-box, the objective function is linear and constraints are known and fixed, but the cost vector need to be predicted from contextual data.


.. autoclass:: pyepo.func.blackboxOpt
    :noindex:
    :members:

``pyepo.func.blackboxOpt`` supports to solve optimization problems in parallel, parameter ``processes`` is the number of processors, 0 for using all available cores. ``lambd`` is a hyperparameter for function smoothing. The range of ``lambd`` should be **10** to **20**.

.. code-block:: python

   import pyepo

   dbb = pyepo.func.blackboxOpt(optmodel, lambd=10, processes=2)


Differentiable Perturbed Optimizer (DPO)
========================================

Differentiable perturbed Optimizer (DPO) [#f3]_ uses Monte-Carlo samples to estimate solutions, in which randomly perturbed costs are sampled to optimize. For the perturbed optimizer, the cost vector needs to be predicted from contextual data and are perturbed with Gaussian noise. The perturbed optimizer is differentiable in its inputs with non-zero Jacobian, thus allowing us to design an algorithm based on stochastic gradient descent.


.. autoclass:: pyepo.func.perturbedOpt
    :noindex:
    :members:

``pyepo.func.perturbedOpt`` supports to solve optimization problems in parallel, parameter ``processes`` is the number of processors, 0 for using all available cores. ``n_samples`` is the number of Monte-Carlo samples to estimate solutions, and ``sigma`` is the variance of Gaussian noise perturbation.

.. code-block:: python

   import pyepo

   dpo = pyepo.func.perturbedOpt(optmodel, n_samples=10, sigma=0.5, processes=2)


Perturbed Fenchel-Young Loss (PYFL)
===================================

Perturbed Fenchel-Young loss (PYFL) function [#f3]_ uses perturbation techniques with Monte-Carlo samples. The use of the loss improves the algorithmic by the specific expression of the gradients of the loss. For the perturbed optimizer, the cost vector need to be predicted from contextual data and are perturbed with Gaussian noise. The Fenchel-Young loss allows to directly optimize a loss between the features and solutions with less computation. Thus, allows us to design an algorithm based on stochastic gradient descent.


.. autoclass:: pyepo.func.perturbedFenchelYoung
    :noindex:
    :members:

``pyepo.func.perturbedFenchelYoung`` supports to solve optimization problems in parallel, parameter ``processes`` is the number of processors, 0 for using all available cores. ``n_samples`` is the number of Monte-Carlo samples to estimate solutions, and ``sigma`` is the variance of Gaussian noise perturbation.

.. code-block:: python

   import pyepo

   pfyl = pyepo.func.perturbedFenchelYoung(optmodel, n_samples=10, sigma=0.5, processes=2)



Noise Contrastive Estimation (NCE)
==================================

NCE Loss function [#f4]_ is a surrogate loss function based on viewing non-optimal solutions as negative examples. For NCE Loss, the constraints are known and fixed, but the cost vector need to be predicted from contextual data. It allows us to design an algorithm based on stochastic gradient descent.

.. autoclass:: pyepo.func.NCE
    :noindex:
    :members:

``pyepo.func.NCE`` supports to solve optimization problems in parallel, parameter ``processes`` is the number of processors, **0 for using all available cores**.

.. code-block:: python

   import pyepo

   nce = pyepo.func.NCE(optmodel, processes=2, solve_ratio=0.05, dataset=dataset_train)


Learning to Rank (LTR)
======================

NCE Loss function [#f5]_ is to learn an objective function that ranks a pool of feasible solutions correctly. For LTR Loss, the constraints are known and fixed, but the cost vector need to be predicted from contextual data. It allows us to design an algorithm based on stochastic gradient descent.

.. autoclass:: pyepo.func.pointwiseLTR
    :noindex:
    :members:

.. autoclass:: pyepo.func.pairwiseLTR
    :noindex:
    :members:

.. autoclass:: pyepo.func.listwiseLTR
    :noindex:
    :members:

``pyepo.func.pointwiseLTR``, ``pyepo.func.pairwiseLTR``, and ``pyepo.func.listwiseLTR`` supports to solve optimization problems in parallel, parameter ``processes`` is the number of processors, **0 for using all available cores**.

.. code-block:: python

   import pyepo

   # pointwise
   ltr = pyepo.func.pointwiseLTR(optmodel, processes=2, solve_ratio=0.05, dataset=dataset_train)
   # pairwise
   ltr = pyepo.func.pairwiseLTR(optmodel, processes=2, solve_ratio=0.05, dataset=dataset_train)
   # listwise
   ltr = pyepo.func.listwiseLTR(optmodel, processes=2, solve_ratio=0.05, dataset=dataset_train)



Parallel Computation
====================

``PyEPO`` supports parallel computation for solving optimization problems in training, where the parameter ``processes`` is the number of processors to be used.

.. warning::  On Windows system, there is missing ``freeze_support`` to run ``multiprocessing`` directly from ``__main__``. When ``processes`` is not 1, try ``if __name__ == "__main__":`` instead of Jupyter notebook or a PY file.

.. rubric:: Footnotes

.. [#f1] Elmachtoub, A. N., & Grigas, P. (2021). Smart “predict, then optimize”. Management Science.
.. [#f2] Vlastelica, M., Paulus, A., Musil, V., Martius, G., & Rolínek, M. (2019). Differentiation of blackbox combinatorial solvers. arXiv preprint arXiv:1912.02175.
.. [#f3] Berthet, Q., Blondel, M., Teboul, O., Cuturi, M., Vert, J. P., & Bach, F. (2020). Learning with differentiable perturbed optimizers. Advances in neural information processing systems, 33, 9508-9519.
.. [#f4] Mulamba, M., Mandi, J., Diligenti, M., Lombardi, M., Bucarey, V., & Guns, T. (2021). Contrastive losses and solution caching for predict-and-optimize. Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence.
.. [#f5] Mandi, J., Bucarey, V., Mulamba, M., & Guns, T. (2022). Decision-focused learning: through the lens of learning to rank. Proceedings of the 39th International Conference on Machine Learning.
