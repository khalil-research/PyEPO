Solution Pool
+++++++++++++

End-to-end predict-then-optimize training involves repeated solving of optimization problems. A solution pool [#f1]_ serves as an inner approximation of the feasible region, reducing computation by reusing previously computed solutions. Instead of solving the full linear/integer program, the method selects the best solution from the cached pool.

The algorithm is illustrated below:

.. image:: ../../images/pool.png
   :width: 500

The solution pool is integrated into all ``pyepo.func`` modules.

``solve_ratio`` controls the fraction of instances solved exactly during training. The default is 1.0 (no caching). When ``solve_ratio`` is less than 1, the pool requires training data via the ``dataset`` parameter to obtain initial solutions.

Example with SPO+ (other functions work the same way):

.. code-block:: python

   import pyepo

   spo = pyepo.func.SPOPlus(optmodel, processes=2, solve_ratio=0.7, dataset=dataset)

.. [#f1] Mulamba, M., Mandi, J., Diligenti, M., Lombardi, M., Bucarey, V., & Guns, T. (2020). Contrastive losses and solution caching for predict-and-optimize. arXiv preprint arXiv:2011.05354.
