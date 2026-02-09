Evaluation
++++++++++


Regret
======

``pyepo.metric.regret`` evaluates the decision quality of a prediction model. Regret is defined as :math:`l_{Regret}(\hat{\mathbf{c}}, \mathbf{c}) = \mathbf{c}^T \mathbf{w}^*(\hat{\mathbf{c}}) - z^*(\mathbf{c})`, which measures the gap between the objective value achieved by the predicted solution and the true optimum.

.. autofunction:: pyepo.metric.regret
    :noindex:

.. code-block:: python

   import pyepo

   regret = pyepo.metric.regret(predmodel, optmodel, testloader)


Unambiguous Regret
==================

When a predicted cost vector :math:`\hat{c}` yields multiple optimal solutions for :math:`\underset{\mathbf{w} \in S}{\min}\;\hat{\mathbf{c}}^T \mathbf{w}`, the unambiguous regret considers the worst case: :math:`l_{URegret}(\hat{\mathbf{c}}, \mathbf{c}) = \underset{\mathbf{w} \in W^*(\mathbf{c})}{\max} \mathbf{w}^T \mathbf{c} - z^*(\mathbf{c})`.

.. image:: ../../images/regret.png
  :width: 650
  :alt: learning curves

As the figure shows, regret and unambiguous regret are nearly identical across all training procedures. While the unambiguous regret is more theoretically rigorous, using it in practice is generally unnecessary.

.. autofunction:: pyepo.metric.unambRegret
    :noindex:

.. code-block:: python

   import pyepo

   regret = pyepo.metric.unambRegret(predmodel, optmodel, testloader)
