Workflow
++++++++

The main ``PyEPO`` workflow has five steps: define an optimization model, build an optimization-aware dataset, choose a training method, train a predictor, and evaluate decision quality.


Core Steps
==========

* **New to PyEPO**: read these pages in order.

  #. :doc:`model` - define the optimization model
  #. :doc:`data` - generate data and build the dataset
  #. :doc:`function` - choose a training method
  #. :doc:`training` - training loop templates
  #. :doc:`evaluation` - decision-quality metrics

* **Want to pick a method**: jump to the *Choosing a Method* section of :doc:`function`. It asks what supervision you have, whether you want a loss or a solution, and whether your problem has special constraints.
* **Training in JAX/Flax**: :doc:`../frontends/jax` follows the PyTorch loss API for ``jax.grad``-based training. MPAX runs natively; non-JAX backends run through ``jax.pure_callback``.
* **Notebooks**: runnable Colab examples are listed in :doc:`../notebooks`.
