Tutorial
++++++++

This guide walks through the main ``PyEPO`` workflow for predict-then-optimize problems: define an optimization model, build an optimization-aware dataset, choose a training module, train a predictor, and evaluate decision quality.


Where to start
==============

* **New to PyEPO** — read the pages below in order. :doc:`examples/model` and :doc:`examples/data` set up the modeling primitives; :doc:`examples/twostage` introduces the regression baseline; :doc:`examples/function` and :doc:`examples/training` cover the end-to-end methods; :doc:`examples/evaluation` defines the metrics.
* **Want to pick a method** — jump to the *Choosing a Method* section of :doc:`examples/function`. It poses three questions (what supervision you have, whether you want a loss or a solution, special constraints) and points you at the right module.
* **Prefer hands-on** — open the `03 Training and Testing <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/03%20Training%20and%20Testing.ipynb>`_ notebook. It compares every method family on the same shortest-path benchmark.


Reference pages
===============

.. toctree::
   :maxdepth: 2

   examples/model
   examples/data
   examples/twostage
   examples/function
   examples/pool
   examples/training
   examples/evaluation
