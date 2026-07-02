Solution Pool
+++++++++++++

End-to-end predict-then-optimize training involves repeated solving of
optimization problems. A solution pool [#f1]_ stores previously computed
solutions and uses them as an inner approximation of the feasible region.

When the pool is used, PyEPO selects the best cached solution under the
predicted cost (lowest objective for minimization, highest for maximization)
instead of solving the original linear/integer program.

Algorithm
=========

**Algorithm**: Gradient descent with inner approximation (numbering follows [#f1]_)

**Input:** :math:`A, b`; training data :math:`\mathcal{D} \equiv \{(x_i, c_i)\}_{i=1}^n`

**Hyperparams:** :math:`\alpha` (learning rate), epochs, :math:`p_{\text{solve}}`

.. math::

   \begin{array}{rl}
   1: & \text{Initialize}\ \omega \\
   2: & \text{Initialize}\ S = \{v^*(c_i) \mid (x_i, c_i) \in \mathcal{D}\} \\
   3: & \textbf{for}\ \text{each epoch}\ \textbf{do} \\
   4: & \quad \textbf{for}\ \text{each instance}\ \textbf{do} \\
   5: & \quad\quad \tilde{c} \leftarrow t(\hat{c})\ \text{with}\ \hat{c} = m(\omega, x) \\
   6: & \quad\quad \textbf{if}\ \mathrm{random}() < p_{\text{solve}}\ \textbf{then} \\
   7: & \quad\quad\quad \text{Obtain}\ v\ \text{by calling a solver for Eq.}\ (1)\ \text{with}\ \tilde{c} \\
   8: & \quad\quad\quad S \leftarrow S \cup \{v\} \\
   9: & \quad\quad \textbf{else} \\
   10: & \quad\quad\quad v = \arg\min\limits_{v' \in S} f(v', \tilde{c}) \\
   11: & \quad\quad \textbf{end if} \\
   12: & \quad\quad \omega \leftarrow \omega - \alpha\, \dfrac{\partial \mathcal{L}^v}{\partial \tilde{c}}\, \dfrac{\partial \tilde{c}}{\partial \omega} \\
   13: & \quad \textbf{end for} \\
   14: & \textbf{end for} \\
   \end{array}

In the algorithm, :math:`\omega` are the predictor weights,
:math:`m(\omega, x) = \hat{c}` is the predicted cost, :math:`t(\cdot)` is an
optional cost transform, :math:`v^*(c)` is the optimum for cost :math:`c`,
:math:`S` is the solution pool, and Eq. (1) is the underlying problem
:math:`\min_{v} \tilde{c}^\top v` over :math:`\{v : A v \le b\}`. The
probability :math:`p_{\text{solve}}` corresponds to ``solve_ratio``.

Usage
=====

Every ``pyepo.func`` module supports the solution pool except ``CaVE``, which
has no pool and repurposes ``solve_ratio`` for its projection branch (see
:doc:`cave`).

``solve_ratio`` sets the probability of solving exactly. PyEPO draws the coin
once per batch rather than per instance, so it is the expected fraction of
batches solved. The default is 1.0 (no caching). When ``solve_ratio`` is less
than 1, pass ``dataset`` to seed the pool with initial solutions; the
contrastive (``NCE`` / ``CMAP``) and learning-to-rank losses require
``dataset`` even at the default.

Example with SPO+ (other functions work the same way):

.. code-block:: python

   import pyepo

   spo = pyepo.func.SPOPlus(optmodel, processes=2, solve_ratio=0.7, dataset=dataset)


Related Pages
=============

* :doc:`../getting_started/function` documents the training methods and their shared options.

.. [#f1] Mulamba, M., Mandi, J., Diligenti, M., Lombardi, M., Bucarey, V., & Guns, T. (2021). Contrastive losses and solution caching for predict-and-optimize. Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence.
