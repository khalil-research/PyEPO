Evaluation
++++++++++


Regret
======

``pyepo.metric.regret`` evaluates the decision quality of a prediction model, usually on a held-out test set. Regret is :math:`l_{Regret}(\hat{\mathbf{c}}, \mathbf{c}) = \mathbf{c}^\top \mathbf{w}^*(\hat{\mathbf{c}}) - \mathbf{c}^\top \mathbf{w}^*(\mathbf{c})`, the excess cost of the predicted solution over the true optimum. By default, instances are aggregated as normalized regret, :math:`\sum_i l_i \, / \, \sum_i |\mathbf{c}_i^\top \mathbf{w}^*(\mathbf{c}_i)|`, which is dimensionless and comparable across problem scales. Use ``reduction`` to switch to ``"sum"``, ``"mean"``, or ``"none"`` (a per-instance array). ``processes`` parallelizes the solves, as in training. Set ``processes=0`` to use all available cores.

.. autofunction:: pyepo.metric.regret
    :noindex:

.. code-block:: python

   import pyepo

   regret = pyepo.metric.regret(predmodel, optmodel, testloader)


Unambiguous Regret
==================

When a predicted cost vector :math:`\hat{\mathbf{c}}` yields multiple optimal solutions for :math:`\underset{\mathbf{w} \in \mathcal{S}}{\min}\;\hat{\mathbf{c}}^\top \mathbf{w}`, the regret depends on which optimum the solver happens to return. The unambiguous regret removes this ambiguity by scoring the worst case: :math:`l_{URegret}(\hat{\mathbf{c}}, \mathbf{c}) = \underset{\mathbf{w} \in W^*(\hat{\mathbf{c}})}{\max} \mathbf{w}^\top \mathbf{c} - \mathbf{c}^\top \mathbf{w}^*(\mathbf{c})`.

.. image:: ../../images/regret.png
  :width: 650
  :alt: learning curves
  :class: light-bg

``unambRegret`` returns only the normalized value. It does not expose a ``reduction`` option. For each instance, it adds a constraint that restricts the feasible region to the tie set and then re-solves for the worst case. This makes it slower than ``regret`` and requires a backend that implements ``addConstr``.

.. autofunction:: pyepo.metric.unambRegret
    :noindex:

.. code-block:: python

   import pyepo

   regret = pyepo.metric.unambRegret(predmodel, optmodel, testloader)
