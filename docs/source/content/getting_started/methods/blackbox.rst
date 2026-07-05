Black-Box Methods
+++++++++++++++++

Black-box methods replace the zero gradient of the discrete solver with a surrogate backward rule. The training loops below define an objective-value loss on the solution returned by the module.

The training loops on this page build on the :ref:`Common Setup <common-setup>` from :doc:`../function`.


Differentiable Black-Box Optimizer (DBB)
========================================

DBB [#f3]_ estimates gradients by comparing the solver output at the predicted cost with the output at a perturbed cost. ``lambd`` is the smoothing hyperparameter.

Given an upstream gradient :math:`\mathbf{d} = \nabla_{\mathbf{w}} \mathcal{L}(\hat{\mathbf{c}}, \cdot) \big|_{\mathbf{w} = \mathbf{w}^*(\hat{\mathbf{c}})}`, DBB approximates the vector-Jacobian product by interpolating the loss between :math:`\hat{\mathbf{c}}` and a perturbed cost :math:`\hat{\mathbf{c}} + \lambda \mathbf{d}`, yielding

.. math::

   \frac{\partial \mathcal{L}(\hat{\mathbf{c}}, \cdot)}{\partial \hat{\mathbf{c}}} \approx \frac{\mathbf{w}^*(\hat{\mathbf{c}} + \lambda \mathbf{d}) - \mathbf{w}^*(\hat{\mathbf{c}})}{\lambda}.

Larger :math:`\lambda` smooths more aggressively. The recommended range is 10 to 20. Unlike SPO+, the resulting surrogate is nonconvex in :math:`\hat{\mathbf{c}}`, which weakens convergence guarantees even when the predictor is convex in its parameters.

.. autoclass:: pyepo.func.DBB
    :noindex:
    :members:

Training loop:

.. code-block:: python

   dbb = pyepo.func.DBB(optmodel, lambd=10, processes=1)
   criterion = nn.L1Loss()

   for epoch in range(20):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           wp = dbb(cp)
           zp = (wp * c).sum(1).view(-1, 1)
           loss = criterion(zp, z)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Negative Identity Backpropagation (NID)
=======================================

NID [#f4]_ treats the solver as a negative identity mapping during backpropagation. It is hyperparameter-free.

NID approximates the solver Jacobian by the (signed) identity, :math:`\partial \mathbf{w}^*(\hat{\mathbf{c}}) / \partial \hat{\mathbf{c}} \approx -\mathbf{I}` for a minimization problem (and :math:`+\mathbf{I}` for maximization). Given an upstream gradient :math:`\mathbf{d} = \partial \mathcal{L}(\hat{\mathbf{c}}, \cdot) / \partial \mathbf{w}^*(\hat{\mathbf{c}})`, the chain rule gives a sign-flipped straight-through estimator,

.. math::

   \frac{\partial \mathcal{L}(\hat{\mathbf{c}}, \cdot)}{\partial \hat{\mathbf{c}}} \approx -\mathbf{d} \quad (\text{minimization}), \qquad \frac{\partial \mathcal{L}(\hat{\mathbf{c}}, \cdot)}{\partial \hat{\mathbf{c}}} \approx \mathbf{d} \quad (\text{maximization}).

For minimization, this rule decreases :math:`\hat{\mathbf{c}}` along directions where :math:`\mathbf{w}^*(\hat{\mathbf{c}})` tends to increase, and vice versa. NID is the special case of DBB in which :math:`\lambda` is chosen so that the interpolated solution coincides with the negative-identity update, but it saves the extra solver call required by DBB.

.. autoclass:: pyepo.func.NID
    :noindex:
    :members:

Training loop:

.. code-block:: python

   nid = pyepo.func.NID(optmodel, processes=1)
   criterion = nn.L1Loss()

   for epoch in range(20):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           wp = nid(cp)
           zp = (wp * c).sum(1).view(-1, 1)
           loss = criterion(zp, z)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


.. rubric:: Footnotes

.. [#f3] Vlastelica, M., Paulus, A., Musil, V., Martius, G., & Rolinek, M. (2020). Differentiation of blackbox combinatorial solvers. In International Conference on Learning Representations.
.. [#f4] Sahoo, S. S., Paulus, A., Vlastelica, M., Musil, V., Kuleshov, V., & Martius, G. (2023). Backpropagation through combinatorial algorithms: Identity with projection works. In International Conference on Learning Representations.
