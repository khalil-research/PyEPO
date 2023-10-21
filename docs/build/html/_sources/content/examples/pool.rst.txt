Solution Pool
+++++++++++++

There is repeated solving during the training of end-to-end predict-then-optimize tasks. Therefore, the solution pool [#f1]_ as an inner approximation of the feasible region is able to reduce computation. With a record of previous solutions, we can argmin over the current solution pool instead of the linear/integer programs.

The corresponding algorithm is as follows:

.. image:: ../../images/pool.png
   :width: 500

The solution pool has already been applied on modules of ``pyepo.func``.

``solve_ratio`` is the ratio of new solutions computed during training, in which the default is 1.0 so the solution pool is not used. When ``solve_ratio`` is less than 1, the solution pool requires training data as ``dataset`` to obtain initial solutions.

Here is an example for SPO+, and other functions are the same.

.. code-block:: python

   import pyepo

   spo = pyepo.func.SPOPlus(optmodel, processes=2, solve_ratio=0.7, dataset=dataset_train)

.. [#f1] Mulamba, M., Mandi, J., Diligenti, M., Lombardi, M., Bucarey, V., & Guns, T. (2020). Contrastive losses and solution caching for predict-and-optimize. arXiv preprint arXiv:2011.05354.
