Contrastive Methods
+++++++++++++++++++

Contrastive methods train against a pool of cached non-optimal solutions, treated as negative examples. ``solve_ratio`` controls how often new instances are solved exactly during training. ``dataset`` seeds the pool and is required by these methods. See :doc:`../../advanced/pool` for details on the solution-pool mechanism.

The training loops on this page build on the :ref:`Common Setup <common-setup>` from :doc:`../function`.


Noise Contrastive Estimation (NCE)
==================================

NCE [#f7]_ is a surrogate loss based on negative examples. It uses a small set of non-optimal solutions as negative samples to maximize the predicted-cost margin between the optimal solution and the rest.

Let :math:`\Gamma` be the cached pool of feasible solutions. For a minimization problem, NCE averages the predicted-cost margin between :math:`\mathbf{w}^*(\mathbf{c})` and every member of the pool,

.. math::

   \mathcal{L}_{\mathrm{NCE}}(\hat{\mathbf{c}}, \mathbf{w}^*(\mathbf{c})) = \frac{1}{|\Gamma|} \sum_{\mathbf{w} \in \Gamma} \big( \hat{\mathbf{c}}^\top \mathbf{w}^*(\mathbf{c}) - \hat{\mathbf{c}}^\top \mathbf{w} \big).

(If :math:`\mathbf{w}^*(\mathbf{c}) \in \Gamma`, its term contributes zero and the sum effectively averages over the negatives.) The gradient has a closed form that requires no solver call in the backward pass,

.. math::

   \frac{\partial \mathcal{L}_{\mathrm{NCE}}(\hat{\mathbf{c}}, \mathbf{w}^*(\mathbf{c}))}{\partial \hat{\mathbf{c}}} = \mathbf{w}^*(\mathbf{c}) - \frac{1}{|\Gamma|} \sum_{\mathbf{w} \in \Gamma} \mathbf{w}.

For a fixed :math:`\Gamma`, this update direction stays constant per instance. ``solve_ratio`` controls how often the pool is refreshed.

.. autoclass:: pyepo.func.NCE
    :noindex:
    :members:

Training loop:

.. code-block:: python

   nce = pyepo.func.NCE(optmodel, processes=2, solve_ratio=0.05, dataset=dataset)

   for epoch in range(20):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           loss = nce(cp, w)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Contrastive MAP (CMAP)
======================

CMAP [#f7]_ is a special case of NCE that uses only the lowest predicted-cost negative sample.

CMAP keeps only the most-violating member of the pool, the one with the smallest predicted-cost objective, yielding a max-margin contrast against the true optimum. For a minimization problem,

.. math::

   \mathcal{L}_{\mathrm{CMAP}}(\hat{\mathbf{c}}, \mathbf{w}^*(\mathbf{c})) = \max_{\mathbf{w} \in \Gamma} \big( \hat{\mathbf{c}}^\top \mathbf{w}^*(\mathbf{c}) - \hat{\mathbf{c}}^\top \mathbf{w} \big) = \hat{\mathbf{c}}^\top \mathbf{w}^*(\mathbf{c}) - \min_{\mathbf{w} \in \Gamma} \hat{\mathbf{c}}^\top \mathbf{w}.

If :math:`\mathbf{w}^*(\mathbf{c}) \in \Gamma`, that entry contributes a zero margin, so the loss is non-negative and vanishes precisely when the predicted costs already make :math:`\mathbf{w}^*(\mathbf{c})` the minimizer over :math:`\Gamma`.

.. autoclass:: pyepo.func.CMAP
    :noindex:
    :members:

Training loop:

.. code-block:: python

   cmap = pyepo.func.CMAP(optmodel, processes=2, solve_ratio=0.05, dataset=dataset)

   for epoch in range(20):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           loss = cmap(cp, w)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


.. rubric:: Footnotes

.. [#f7] Mulamba, M., Mandi, J., Diligenti, M., Lombardi, M., Bucarey, V., & Guns, T. (2021). Contrastive losses and solution caching for predict-and-optimize. Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence.
