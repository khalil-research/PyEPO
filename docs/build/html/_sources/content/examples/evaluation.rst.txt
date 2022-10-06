Evaluation
++++++++++


Regret
======

``pyepo.eval`` is to evaluate model performance. Regret :math:`l_{SPO}(\hat{c}, c) = c^T w^*(\hat{c}) - z^*(c)` aims to measure the error in decision-making. It evaluates the distance between the objective value of the solution from predicted cost :math:`\hat{c}` and the true optimal objective value :math:`z^*(c)`.

.. autofunction:: pyepo.metric.regret

.. code-block:: python

   import pyepo

   regret = pyepo.metric.regret(predmodel, optmodel, testloader)


Unambiguous Regret
==================

Given a cost vector :math:`\hat{c}`, there may be multiple optimal solutions of :math:`\underset{w \in S}{\min}\;\hat{c}^T w`. Unambiguous Regret Loss :math:`l_{URegret}(\hat{c}, c) = \underset{w \in W^*(c)}{\max} w^T c - z^*(c)` considers the worst cases.

However, the regret and the unambiguous regret are almost same in all training procedures. Therefore, although the unambiguous regret is more theoretically rigorous, it is not necessary to consider it in practice.

.. autofunction:: pyepo.metric.unambRegret

.. code-block:: python

   import pyepo

   regret = pyepo.metric.unambRegret(predmodel, optmodel, testloader)
