Regularized Methods
+++++++++++++++++++

A pure linear optimization layer is not differentiable: the optimal solution :math:`\mathbf{w}^*(\hat{\mathbf{c}}) = \arg\min_{\mathbf{w} \in \mathcal{S}} \hat{\mathbf{c}}^\top \mathbf{w}` is piecewise constant in :math:`\hat{\mathbf{c}}`. It jumps between vertices of :math:`\mathcal{S}`, and its derivative is zero almost everywhere. Regularized methods fix this by adding a strongly convex regularizer :math:`\Omega` to the linear objective, yielding a regularized solution

.. math::

   \hat{\mathbf{w}}_\Omega(\hat{\mathbf{c}}) = \arg\min_{\mathbf{w} \in \mathrm{conv}(\mathcal{S})} \hat{\mathbf{c}}^\top \mathbf{w} + \Omega(\mathbf{w}),

which is unique, lies inside :math:`\mathrm{conv}(\mathcal{S})`, and varies continuously with :math:`\hat{\mathbf{c}}`. This deterministic smoothing is dual to the stochastic smoothing of the perturbed methods: both replace the piecewise-constant solution map with a continuous surrogate, the former via a convex regularizer and the latter via Monte Carlo averaging over a noise distribution [#f6]_.

PyEPO implements the L2 special case :math:`\Omega(\mathbf{w}) = \tfrac{\lambda}{2} \|\mathbf{w}\|_2^2` and solves the regularized program with batched Frank-Wolfe iteration. Frank-Wolfe accesses :math:`\mathrm{conv}(\mathcal{S})` through the original LP/ILP solver for :math:`\mathcal{S}`, used as a *linear minimization oracle*. ``lambd`` is the L2 strength :math:`\lambda`. ``max_iter`` caps Frank-Wolfe iterations, and ``tol`` is the convergence tolerance.

The training loops on this page build on the :ref:`Common Setup <common-setup>` from :doc:`../function`.


L2 Regularized Frank-Wolfe Optimizer (RFWO)
===========================================

RFWO [#f6]_ exposes :math:`\hat{\mathbf{w}}_\lambda(\hat{\mathbf{c}})` directly as a differentiable layer. Since it returns a regularized solution rather than a loss, the user defines a task loss on the output. A common choice is MSE against :math:`\mathbf{w}^*(\mathbf{c})`.

.. math::

   \hat{\mathbf{w}}_\lambda(\hat{\mathbf{c}}) = \arg\min_{\mathbf{w} \in \mathrm{conv}(\mathcal{S})} \hat{\mathbf{c}}^\top \mathbf{w} + \tfrac{\lambda}{2} \|\mathbf{w}\|_2^2.

.. autoclass:: pyepo.func.RFWO
    :noindex:
    :members:

Training loop:

.. code-block:: python

   rfwo = pyepo.func.RFWO(optmodel, lambd=1.0, processes=1)
   criterion = nn.MSELoss()

   for epoch in range(20):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           wr = rfwo(cp)
           loss = criterion(wr, w)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


L2 Regularized Frank-Wolfe with Fenchel-Young Loss (RFYL)
=========================================================

RFYL [#f6]_ pairs the RFWO layer with the **Fenchel-Young loss** [#f13]_ of the regularizer :math:`\Omega`, returning a scalar loss that compares the predicted cost :math:`\hat{\mathbf{c}}` to the true optimum :math:`\mathbf{w}^*(\mathbf{c})`.

For any convex regularizer :math:`\Omega: \mathrm{conv}(\mathcal{S}) \to \mathbb{R}` with conjugate :math:`\Omega^*(\boldsymbol{\theta}) = \sup_{\mathbf{w}} \boldsymbol{\theta}^\top \mathbf{w} - \Omega(\mathbf{w})`, the Fenchel-Young loss is

.. math::

   \mathcal{L}_\Omega^{\mathrm{FY}}(\hat{\mathbf{c}}, \mathbf{w}^*(\mathbf{c})) = \Omega(\mathbf{w}^*(\mathbf{c})) + \Omega^*(-\hat{\mathbf{c}}) + \hat{\mathbf{c}}^\top \mathbf{w}^*(\mathbf{c}).

The Fenchel-Young inequality gives three properties used by the training loss:

1. **Non-negative**, with equality iff :math:`\mathbf{w}^*(\mathbf{c}) = \hat{\mathbf{w}}_\Omega(\hat{\mathbf{c}})`.
2. **Convex** in :math:`\hat{\mathbf{c}}`.
3. By Danskin's theorem, the gradient is the residual between the regularized prediction and the target,

   .. math::

      \frac{\partial \mathcal{L}_\Omega^{\mathrm{FY}}(\hat{\mathbf{c}}, \mathbf{w}^*(\mathbf{c}))}{\partial \hat{\mathbf{c}}} = \mathbf{w}^*(\mathbf{c}) - \hat{\mathbf{w}}_\Omega(\hat{\mathbf{c}}),

   without differentiating through the Frank-Wolfe iterate.

PyEPO specializes this to :math:`\Omega(\mathbf{w}) = \tfrac{\lambda}{2}\|\mathbf{w}\|_2^2`. Substituting the closed form of :math:`\Omega^*` and the definition of :math:`\hat{\mathbf{w}}_\lambda` collapses the loss to

.. math::

   \mathcal{L}_{\mathrm{RFYL}}(\hat{\mathbf{c}}, \mathbf{w}^*(\mathbf{c})) = \Omega(\mathbf{w}^*(\mathbf{c})) - \Omega(\hat{\mathbf{w}}_\lambda(\hat{\mathbf{c}})) + \hat{\mathbf{c}}^\top \big( \mathbf{w}^*(\mathbf{c}) - \hat{\mathbf{w}}_\lambda(\hat{\mathbf{c}}) \big),

a "compare regularizers + linear residual" form. The gradient remains :math:`\mathbf{w}^*(\mathbf{c}) - \hat{\mathbf{w}}_\lambda(\hat{\mathbf{c}})`.

.. autoclass:: pyepo.func.RFY
    :noindex:
    :members:

Training loop:

.. code-block:: python

   rfyl = pyepo.func.RFY(optmodel, lambd=1.0, processes=1)

   for epoch in range(20):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           loss = rfyl(cp, w)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


.. rubric:: Footnotes

.. [#f6] Dalle, G., Baty, L., Bouvier, L., & Parmentier, A. (2022). Learning with Combinatorial Optimization Layers: a Probabilistic Approach. arXiv preprint arXiv:2207.13513.
.. [#f13] Blondel, M., Martins, A. F. T., & Niculae, V. (2020). Learning with Fenchel-Young losses. Journal of Machine Learning Research, 21(35), 1-69.
