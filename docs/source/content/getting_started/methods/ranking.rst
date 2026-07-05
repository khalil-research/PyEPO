Learning to Rank
++++++++++++++++

Learning to rank [#f8]_ treats predict-then-optimize training as ranking a pool of feasible solutions. Predicted costs assign scores to solutions, and the loss encourages the optimal solution to rank highest. Each variant scores the ranking differently. Like contrastive methods, LTR uses ``solve_ratio`` and ``dataset`` to manage the pool.

The training loops on this page build on the :ref:`Common Setup <common-setup>` from :doc:`../function`.

* **Pointwise** regresses each predicted score :math:`\hat{\mathbf{c}}^\top \mathbf{w}` toward the true value :math:`\mathbf{c}^\top \mathbf{w}` for every :math:`\mathbf{w} \in \Gamma`.
* **Pairwise** enforces a margin between the optimal solution and each suboptimal one.
* **Listwise** models the entire ranking distribution with a SoftMax over the negative predicted costs (for a minimization problem, so the lowest-cost solution gets the highest probability),

  .. math::

     P(\mathbf{w}' \mid \hat{\mathbf{c}}) = \frac{\exp(-\hat{\mathbf{c}}^\top \mathbf{w}')}{\sum_{\mathbf{w} \in \Gamma} \exp(-\hat{\mathbf{c}}^\top \mathbf{w})},

  and minimizes the cross-entropy against the true ranking distribution,

  .. math::

     \mathcal{L}_{\mathrm{LTR}}^{\mathrm{list}}(\hat{\mathbf{c}}, \mathbf{c}) = - \sum_{\mathbf{w} \in \Gamma} P(\mathbf{w} \mid \mathbf{c}) \log P(\mathbf{w} \mid \hat{\mathbf{c}}).

  The gradient has a closed form,

  .. math::

     \frac{\partial \mathcal{L}_{\mathrm{LTR}}^{\mathrm{list}}(\hat{\mathbf{c}}, \mathbf{c})}{\partial \hat{\mathbf{c}}} = \sum_{\mathbf{w} \in \Gamma} P(\mathbf{w} \mid \mathbf{c})\, \mathbf{w} - \sum_{\mathbf{w} \in \Gamma} P(\mathbf{w} \mid \hat{\mathbf{c}})\, \mathbf{w}.


Pointwise LTR
=============

.. autoclass:: pyepo.func.ptLTR
    :noindex:
    :members:

Training loop:

.. code-block:: python

   ltr = pyepo.func.ptLTR(optmodel, processes=1, solve_ratio=0.05, dataset=dataset)

   for epoch in range(20):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           loss = ltr(cp, c)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Pairwise LTR
============

.. autoclass:: pyepo.func.prLTR
    :noindex:
    :members:

Training loop:

.. code-block:: python

   ltr = pyepo.func.prLTR(optmodel, processes=1, solve_ratio=0.05, dataset=dataset)

   for epoch in range(20):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           loss = ltr(cp, c)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Listwise LTR
============

.. autoclass:: pyepo.func.lsLTR
    :noindex:
    :members:

Training loop:

.. code-block:: python

   ltr = pyepo.func.lsLTR(optmodel, processes=1, solve_ratio=0.05, dataset=dataset)

   for epoch in range(20):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           loss = ltr(cp, c)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


.. rubric:: Footnotes

.. [#f8] Mandi, J., Bucarey, V., Mulamba, M., & Guns, T. (2022). Decision-focused learning: through the lens of learning to rank. Proceedings of the 39th International Conference on Machine Learning.
