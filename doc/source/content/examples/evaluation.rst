Evaluation
++++++++++


True SPO
========

``pyepo.eval`` is to evaluate model performance. SPO loss :math:`l_{SPO}(\hat{c}, c) = c^T w^*(\hat{c}) - z^*(c)` aims to measure the error in decision-making. It evaluates the distance between the objective value of the solution from predicted cost :math:`\hat{c}` and the true optimal objective value :math:`z^*(c)`.

.. autofunction:: pyepo.eval.trueSPO


Unambiguous SPO
===============

Given a cost vector :math:`\hat{c}`, there may be multiple optimal solutions of :math:`\underset{w \in S}{\min}\;\hat{c}^T w`. Unambiguous SPO Loss :math:`l_{USPO}(\hat{c}, c) = \underset{w \in W^*(c)}{\max} w^T c - z^*(c)` considers the worst cases.

.. autofunction:: pyepo.eval.unambSPO
