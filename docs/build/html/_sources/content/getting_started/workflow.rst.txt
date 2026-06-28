Workflow
++++++++

The main ``PyEPO`` workflow has four steps: define an optimization model, build an optimization-aware dataset, choose and train with a PyEPO method, and evaluate decision quality.


Core Steps
==========

* **New to PyEPO**: read these pages in order.

  #. :doc:`model` - define the optimization model
  #. :doc:`data` - generate data and build the dataset
  #. :doc:`function` - choose a training method and train the predictor
  #. :doc:`evaluation` - decision-quality metrics

* **Want to pick a method**: the *Choosing a Method* section of :doc:`function` groups the methods by whether they return a loss or a solution, with a summary table of return types and inputs.
* **Training in JAX/Flax**: :doc:`../frontends/jax` follows the PyTorch loss API for ``jax.grad``-based training. MPAX runs natively; non-JAX backends run through ``jax.pure_callback``.
* **Notebooks**: runnable Colab examples are listed in :doc:`../notebooks`.
