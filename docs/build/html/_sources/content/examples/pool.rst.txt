Solution Pool
+++++++++++++

End-to-end predict-then-optimize training involves repeated solving of optimization problems. A solution pool [#f1]_ serves as an inner approximation of the feasible region, reducing computation by reusing previously computed solutions. Instead of solving the full linear/integer program, PyEPO selects the best cached solution from the pool.

The algorithm is shown below.

**Algorithm** — Gradient descent with inner approximation (numbering follows [#f1]_)

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

The solution pool is integrated into all ``pyepo.func`` modules.

``solve_ratio`` controls the fraction of instances solved exactly during training. The default is 1.0 (no caching). When ``solve_ratio`` is less than 1, pass ``dataset`` to seed the pool with initial solutions.

Example with SPO+ (other functions work the same way):

.. code-block:: python

   import pyepo

   spo = pyepo.func.SPOPlus(optmodel, processes=2, solve_ratio=0.7, dataset=dataset)

.. [#f1] Mulamba, M., Mandi, J., Diligenti, M., Lombardi, M., Bucarey, V., & Guns, T. (2020). Contrastive losses and solution caching for predict-and-optimize. arXiv preprint arXiv:2011.05354.
