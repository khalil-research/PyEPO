Surrogate Losses
++++++++++++++++

Surrogate losses replace the non-differentiable regret with a differentiable training objective.

The training loops on this page build on the :ref:`Common Setup <common-setup>` from :doc:`../function`.


Smart Predict-then-Optimize+ Loss (SPO+)
========================================

SPO+ [#f1]_ is a convex surrogate for the SPO loss, also called regret: the decision error of the downstream optimization.

The derivation starts from the observation that for any :math:`\alpha \geq 0`,

.. math::

   \mathrm{regret}(\hat{\mathbf{c}}, \mathbf{c}) \leq \max_{\mathbf{w} \in \mathcal{S}} \big\{ \mathbf{c}^\top \mathbf{w} - \alpha\, \hat{\mathbf{c}}^\top \mathbf{w} \big\} + z^*(\alpha \hat{\mathbf{c}}) - z^*(\mathbf{c}).

Set :math:`\alpha = 2`. Because :math:`z^*` is a minimum and :math:`\mathbf{w}^*(\mathbf{c})` is feasible, :math:`z^*(2 \hat{\mathbf{c}}) \leq 2 \hat{\mathbf{c}}^\top \mathbf{w}^*(\mathbf{c})`. This gives the SPO+ loss,

.. math::

   \mathcal{L}_{\mathrm{SPO+}}(\hat{\mathbf{c}}, \mathbf{c}) = -z^*(2 \hat{\mathbf{c}} - \mathbf{c}) + 2\, \hat{\mathbf{c}}^\top \mathbf{w}^*(\mathbf{c}) - z^*(\mathbf{c}),

which is convex in :math:`\hat{\mathbf{c}}` and upper-bounds the regret. By Danskin's theorem, a subgradient is

.. math::

   2 \big( \mathbf{w}^*(\mathbf{c}) - \mathbf{w}^*(2 \hat{\mathbf{c}} - \mathbf{c}) \big) \in \frac{\partial \mathcal{L}_{\mathrm{SPO+}}(\hat{\mathbf{c}}, \mathbf{c})}{\partial \hat{\mathbf{c}}}.

.. autoclass:: pyepo.func.SPOPlus
    :noindex:
    :members:

``processes`` sets the number of solver processes (``0`` uses all available cores).

Training loop:

.. code-block:: python

   spo = pyepo.func.SPOPlus(optmodel, processes=1)

   for epoch in range(20):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           loss = spo(cp, c, w, z)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Perturbation Gradient (PG)
==========================

PG [#f11]_ is a surrogate loss based on zeroth-order gradient approximation.

By Danskin's theorem, regret can be written as the directional derivative of :math:`z^*(\hat{\mathbf{c}})` along :math:`\mathbf{c}` (up to a constant independent of :math:`\hat{\mathbf{c}}`). PG approximates this directional derivative with finite differences. Two variants are available, backward (``two_sides=False``) and central (``two_sides=True``):

.. math::

   \mathcal{L}_{\mathrm{PG}}^{\mathrm{back}}(\hat{\mathbf{c}}, \mathbf{c}) &= \frac{z^*(\hat{\mathbf{c}}) - z^*(\hat{\mathbf{c}} - \sigma \mathbf{c})}{\sigma}, \\
   \mathcal{L}_{\mathrm{PG}}^{\mathrm{cent}}(\hat{\mathbf{c}}, \mathbf{c}) &= \frac{z^*(\hat{\mathbf{c}} + \sigma \mathbf{c}) - z^*(\hat{\mathbf{c}} - \sigma \mathbf{c})}{2 \sigma},

where :math:`\sigma > 0` is the finite-difference width (the ``sigma`` parameter). The corresponding gradient estimate is

.. math::

   \frac{\partial \mathcal{L}_{\mathrm{PG}}^{\mathrm{back}}(\hat{\mathbf{c}}, \mathbf{c})}{\partial \hat{\mathbf{c}}} \approx \frac{\mathbf{w}^*(\hat{\mathbf{c}}) - \mathbf{w}^*(\hat{\mathbf{c}} - \sigma \mathbf{c})}{\sigma}.

.. autoclass:: pyepo.func.PG
    :noindex:
    :members:

``sigma`` controls the finite-difference width. ``two_sides`` enables central differencing.

Training loop:

.. code-block:: python

   pg = pyepo.func.PG(optmodel, sigma=0.1, two_sides=False, processes=1)

   for epoch in range(20):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           loss = pg(cp, c)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


.. rubric:: Footnotes

.. [#f1] Elmachtoub, A. N., & Grigas, P. (2021). Smart "predict, then optimize". Management Science.
.. [#f11] Gupta, V., & Huang, M. (2024). Decision-Focused Learning with Directional Gradients. Advances in Neural Information Processing Systems, 37.
