Perturbed Methods
+++++++++++++++++

Perturbed methods estimate gradients by Monte Carlo averaging over random perturbations of the cost vector. They share two design knobs: ``n_samples`` (number of Monte Carlo samples) and ``sigma`` (perturbation amplitude).

The training loops on this page build on the :ref:`Common Setup <common-setup>` from :doc:`../function`.


Differentiable Perturbed Optimizer (DPO)
========================================

DPO [#f5]_ uses Monte Carlo sampling to estimate solutions by optimizing randomly perturbed costs. Its custom backward pass provides a gradient estimator for end-to-end training. ``DPO`` is the additive Gaussian version; ``DPOMul`` is the multiplicative version for sign-sensitive oracles [#f6]_. The multiplicative variant assumes predicted costs already have the intended nonzero sign. For nonnegative-cost problems, use a positive-output predictor such as ``nn.Softplus()`` with a small epsilon.

DPO replaces the piecewise-constant solution map :math:`\hat{\mathbf{c}} \mapsto \mathbf{w}^*(\hat{\mathbf{c}})` with the expectation over a random perturbation,

.. math::

   \mathbb{E}_{\boldsymbol{\xi}} \big[ \mathbf{w}^*(\hat{\mathbf{c}} + \sigma \boldsymbol{\xi}) \big] \approx \frac{1}{K} \sum_{\kappa=1}^{K} \mathbf{w}^*(\hat{\mathbf{c}} + \sigma \boldsymbol{\xi}_\kappa),

where :math:`\boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})` and :math:`K` is ``n_samples``.

The multiplicative variant ``DPOMul`` perturbs each entry of :math:`\hat{\mathbf{c}}` as

.. math::

   \hat{\mathbf{c}} \odot \exp\!\big(\sigma \boldsymbol{\xi} - \tfrac{1}{2} \sigma^2\big),

with :math:`\odot` the Hadamard product. The exponential factor preserves the sign of each cost entry, which is required when the solver expects, e.g., nonnegative edge costs.

.. autoclass:: pyepo.func.DPO
    :noindex:
    :members:

.. autoclass:: pyepo.func.DPOMul
    :noindex:
    :members:

Training loop (swap ``predmodel`` for ``positive_predmodel`` when using ``DPOMul``):

.. code-block:: python

   dpo = pyepo.func.DPO(optmodel, n_samples=10, sigma=0.5, processes=2)
   # if using DPOMul, replace predmodel with positive_predmodel below
   # dpo = pyepo.func.DPOMul(optmodel, n_samples=10, sigma=0.5, processes=2)

   criterion = nn.MSELoss()

   for epoch in range(20):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           we = dpo(cp)
           loss = criterion(we, w)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Perturbed Fenchel-Young Loss (PFYL)
===================================

PFYL [#f5]_ uses the same perturbed expected solution as DPO inside a Fenchel-Young loss and compares it directly with the true optimal solution. It returns a scalar loss. ``PFY`` is the additive Gaussian version; ``PFYMul`` is the multiplicative sign-preserving variant with the same sign convention as ``DPOMul``.

Let :math:`F(\hat{\mathbf{c}}) = \mathbb{E}_{\boldsymbol{\xi}}\big[ \min_{\mathbf{w} \in \mathcal{S}} (\hat{\mathbf{c}} + \sigma \boldsymbol{\xi})^\top \mathbf{w} \big]` be the expected perturbed minimum, and let :math:`\Omega` be its Fenchel conjugate. The perturbed Fenchel-Young loss is

.. math::

   \mathcal{L}_{\mathrm{PFYL}}(\hat{\mathbf{c}}, \mathbf{w}^*(\mathbf{c})) = \hat{\mathbf{c}}^\top \mathbf{w}^*(\mathbf{c}) - F(\hat{\mathbf{c}}) - \Omega(\mathbf{w}^*(\mathbf{c})).

The dual term :math:`\Omega(\mathbf{w}^*(\mathbf{c}))` does not depend on :math:`\hat{\mathbf{c}}`, so the gradient collapses to a simple difference of solutions,

.. math::

   \frac{\partial \mathcal{L}_{\mathrm{PFYL}}(\hat{\mathbf{c}}, \mathbf{w}^*(\mathbf{c}))}{\partial \hat{\mathbf{c}}} = \mathbf{w}^*(\mathbf{c}) - \mathbb{E}_{\boldsymbol{\xi}} \big[\mathbf{w}^*(\hat{\mathbf{c}} + \sigma \boldsymbol{\xi})\big] \approx \mathbf{w}^*(\mathbf{c}) - \frac{1}{K} \sum_{\kappa=1}^{K} \mathbf{w}^*(\hat{\mathbf{c}} + \sigma \boldsymbol{\xi}_\kappa).

The multiplicative variant uses the same sign-preserving perturbation,

.. math::

   \hat{\mathbf{c}} \odot \exp\!\big(\sigma \boldsymbol{\xi} - \tfrac{1}{2} \sigma^2\big).

.. autoclass:: pyepo.func.PFY
    :noindex:
    :members:

.. autoclass:: pyepo.func.PFYMul
    :noindex:
    :members:

Training loop (swap ``predmodel`` for ``positive_predmodel`` when using ``PFYMul``):

.. code-block:: python

   pfy = pyepo.func.PFY(optmodel, n_samples=10, sigma=0.5, processes=2)
   # if using PFYMul, replace predmodel with positive_predmodel below
   # pfy = pyepo.func.PFYMul(optmodel, n_samples=10, sigma=0.5, processes=2)

   for epoch in range(20):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           loss = pfy(cp, w)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Implicit Maximum Likelihood Estimator (I-MLE)
=============================================

I-MLE [#f9]_ uses the perturb-and-MAP framework. It samples noise from a Sum-of-Gamma distribution and interpolates the loss function to approximate finite differences. ``lambd`` controls the interpolation step.

I-MLE is framed as imitation learning: bring the model distribution :math:`p(\mathbf{w} \mid \hat{\mathbf{c}})` closer to a target distribution :math:`q(\mathbf{w} \mid \hat{\mathbf{c}})` by minimizing their KL divergence. An upstream task gradient :math:`\mathbf{d} = \nabla_{\mathbf{w}} \mathcal{L}(\hat{\mathbf{c}}, \cdot) \big|_{\mathbf{w} = \mathbf{w}^*(\hat{\mathbf{c}})}` induces a virtual update :math:`\hat{\mathbf{c}}' = \hat{\mathbf{c}} + \lambda \mathbf{d}`, and the gradient is estimated by a directional finite difference between the smoothed solutions at :math:`\hat{\mathbf{c}}'` and :math:`\hat{\mathbf{c}}`:

.. math::

   \frac{\partial \mathcal{L}_{\mathrm{IMLE}}(\hat{\mathbf{c}}, \cdot)}{\partial \hat{\mathbf{c}}} \approx \frac{1}{K \lambda} \sum_{\kappa=1}^{K} \Big( \mathbf{w}^*(\hat{\mathbf{c}} + \lambda \mathbf{d} + \sigma \boldsymbol{\xi}_\kappa) - \mathbf{w}^*(\hat{\mathbf{c}} + \sigma \boldsymbol{\xi}_\kappa) \Big),

where :math:`\sigma` smooths the solution map and :math:`\lambda` (``lambd``) is the finite-difference step. The same noise realization :math:`\boldsymbol{\xi}_\kappa` is shared across the two perturbed evaluations to reduce variance.

.. autoclass:: pyepo.func.IMLE
    :noindex:
    :members:

Training loop:

.. code-block:: python

   imle = pyepo.func.IMLE(optmodel, n_samples=10, sigma=1.0, lambd=10, processes=2)
   criterion = nn.L1Loss()

   for epoch in range(20):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           we = imle(cp)
           loss = criterion(we, w)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Adaptive Implicit Maximum Likelihood Estimator (AI-MLE)
=======================================================

AI-MLE [#f10]_ extends I-MLE with an adaptive interpolation step.

AI-MLE uses the same finite-difference estimator as I-MLE, but replaces the fixed step size :math:`\lambda` with an adaptive choice driven by the magnitudes of the predicted cost and the upstream gradient,

.. math::

   \lambda_t = \alpha_t \cdot \frac{\|\hat{\mathbf{c}}\|_2}{\|\mathbf{d}\|_2},

where :math:`\mathbf{d}` is the upstream task gradient and :math:`\alpha_t > 0` is tuned online. When an exponential moving average of the fraction of nonzero gradient entries drops below one, :math:`\alpha_t` is increased; otherwise it is decreased. This rescaling keeps the perturbation magnitude commensurate with :math:`\hat{\mathbf{c}}` and decouples the step from the absolute scale of :math:`\mathbf{d}`, removing the need to tune :math:`\lambda` by hand.

.. autoclass:: pyepo.func.AIMLE
    :noindex:
    :members:

Training loop (the adaptive step works with far fewer samples than I-MLE):

.. code-block:: python

   aimle = pyepo.func.AIMLE(optmodel, n_samples=2, sigma=1.0, processes=2)
   criterion = nn.L1Loss()

   for epoch in range(20):
       for x, c, w, z in dataloader:
           cp = predmodel(x)
           we = aimle(cp)
           loss = criterion(we, w)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


.. rubric:: Footnotes

.. [#f5] Berthet, Q., Blondel, M., Teboul, O., Cuturi, M., Vert, J. P., & Bach, F. (2020). Learning with differentiable perturbed optimizers. Advances in Neural Information Processing Systems, 33, 9508-9519.
.. [#f6] Dalle, G., Baty, L., Bouvier, L., & Parmentier, A. (2022). Learning with Combinatorial Optimization Layers: a Probabilistic Approach. arXiv preprint arXiv:2207.13513.
.. [#f9] Niepert, M., Minervini, P., & Franceschi, L. (2021). Implicit MLE: backpropagating through discrete exponential family distributions. Advances in Neural Information Processing Systems, 34, 14567-14579.
.. [#f10] Minervini, P., Franceschi, L., & Niepert, M. (2023). Adaptive perturbation-based gradient estimation for discrete latent variable models. Proceedings of the AAAI Conference on Artificial Intelligence.
