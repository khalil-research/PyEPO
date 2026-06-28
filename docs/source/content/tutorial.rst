Tutorial
++++++++

This guide walks through the main ``PyEPO`` workflow for predict-then-optimize problems: define an optimization model, build an optimization-aware dataset, choose a training module, train a predictor, and evaluate decision quality.


Where to Start
==============

* **New to PyEPO**: read the pages below in order.

  #. :doc:`examples/model` — define the optimization model
  #. :doc:`examples/data` — generate data and build the dataset
  #. :doc:`examples/twostage` — the two-stage regression baseline
  #. :doc:`examples/function` — the end-to-end training modules
  #. :doc:`examples/training` — training loop templates
  #. :doc:`examples/pool` — the cached solution pool shared by the contrastive and learning-to-rank methods
  #. :doc:`examples/evaluation` — decision-quality metrics

* **Want to pick a method**: jump to the *Choosing a Method* section of :doc:`examples/function`. It poses three questions (what supervision you have, whether you want a loss or a solution, special constraints) and points you at the right module.
* **Training in JAX/Flax**: :doc:`examples/jax` mirrors every loss for ``jax.grad``-based end-to-end training. MPAX runs natively; non-JAX backends run through ``jax.pure_callback``.
* **Prefer hands-on**: jump to the *Notebooks* section below for runnable Colab examples grouped by purpose.


Notebooks
=========

Each notebook is a self-contained Colab walkthrough. They are grouped by purpose, and most entries link back to the docs section that explains the underlying API.

Getting Started
---------------

* `01 Optimization Model <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/01%20Optimization%20Model.ipynb>`_: build an ``optModel`` from a GurobiPy / COPT / Pyomo / OR-Tools / MPAX backend. Pairs with :doc:`examples/model`.
* `02 Optimization Dataset <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/02%20Optimization%20Dataset.ipynb>`_: generate synthetic data and wrap it in ``optDataset``. Pairs with :doc:`examples/data`.
* `03 Training and Testing <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/03%20Training%20and%20Testing.ipynb>`_: train and compare every method family on the same shortest-path benchmark. Pairs with :doc:`examples/training` and :doc:`examples/evaluation`.

Method Deep Dives
-----------------

* `04 CaVE for Binary Linear Programs <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/04%20CaVE%20for%20Binary%20Linear%20Programs.ipynb>`_: train with the cone-aligned CaVE loss on TSP and compare against SPO+. Pairs with the *Cone-Aligned Estimation* section of :doc:`examples/function`.
* `08 kNN Robust Losses <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/08%20kNN%20Robust%20Losses.ipynb>`_: train with the kNN robust loss via ``optDatasetKNN``. Pairs with the *optDatasetKNN* section of :doc:`examples/data`.

GPU Acceleration
----------------

* `09 Solving on MPAX with PDHG <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/09%20Solving%20on%20MPAX%20with%20PDHG.ipynb>`_: batch-solve LPs on GPU via MPAX, end-to-end without CPU round-trips. See the MPAX backend (``optMpaxModel``) in the *Solver Backend Subclass* section of :doc:`examples/model`.
* `10 JAX Frontend <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/10%20JAX%20Frontend.ipynb>`_: train PyEPO losses in JAX/Flax with ``jax.grad``. MPAX is GPU-native and jittable; non-JAX backends run through ``jax.pure_callback``. Pairs with :doc:`examples/jax`.

Applied Examples
----------------

* `05 2D Knapsack Solution Visualization <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/05%202D%20Knapsack%20Solution%20Visualization.ipynb>`_: visualize selected items for a 2D knapsack instance to inspect what the trained predictor is doing.
* `06 Warcraft Shortest Path <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/06%20Warcraft%20Shortest%20Path.ipynb>`_: train an image-based shortest-path predictor on the Warcraft terrain dataset, using a CNN encoder feeding into PyEPO.
* `07 Real-World Energy Scheduling <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/07%20Real-World%20Energy%20Scheduling.ipynb>`_: apply PyEPO to a real-world energy scheduling benchmark with measured demand data.


Reference Pages
===============

.. toctree::
   :maxdepth: 2

   examples/model
   examples/data
   examples/twostage
   examples/function
   examples/jax
   examples/training
   examples/pool
   examples/evaluation
