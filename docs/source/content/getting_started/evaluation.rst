Evaluation
++++++++++


Regret
======

``pyepo.metric.regret`` evaluates the decision quality of a prediction model, usually on a held-out test set. Regret is defined as :math:`l_{Regret}(\hat{\mathbf{c}}, \mathbf{c}) = \mathbf{c}^\top \mathbf{w}^*(\hat{\mathbf{c}}) - \mathbf{c}^\top \mathbf{w}^*(\mathbf{c})`, which measures the excess cost of the predicted solution over the true optimum. By default the instances are aggregated as the normalized regret :math:`\sum_i l_i \, / \, \sum_i |\mathbf{c}_i^\top \mathbf{w}^*(\mathbf{c}_i)|`, dimensionless and comparable across problem scales; ``reduction`` switches to ``"sum"``, ``"mean"``, or ``"none"`` (per-instance array). ``processes`` parallelizes the solving, as in training (``0`` uses all available cores).

.. autofunction:: pyepo.metric.regret
    :noindex:

.. code-block:: python

   import pyepo

   regret = pyepo.metric.regret(predmodel, optmodel, testloader)


Unambiguous Regret
==================

When a predicted cost vector :math:`\hat{\mathbf{c}}` yields multiple optimal solutions for :math:`\underset{\mathbf{w} \in S}{\min}\;\hat{\mathbf{c}}^T \mathbf{w}`, the regret depends on which optimum the solver happens to return. The unambiguous regret removes this ambiguity by scoring the worst case: :math:`l_{URegret}(\hat{\mathbf{c}}, \mathbf{c}) = \underset{\mathbf{w} \in W^*(\hat{\mathbf{c}})}{\max} \mathbf{w}^\top \mathbf{c} - \mathbf{c}^\top \mathbf{w}^*(\mathbf{c})`.

.. image:: ../../images/regret.png
  :width: 650
  :alt: learning curves
  :class: light-bg

``unambRegret`` returns only the normalized value (no ``reduction`` option). It enumerates the tie set by re-solving each instance with added constraints, so it is slower than ``regret`` and needs a backend that implements ``addConstr``.

.. autofunction:: pyepo.metric.unambRegret
    :noindex:

.. code-block:: python

   import pyepo

   regret = pyepo.metric.unambRegret(predmodel, optmodel, testloader)
