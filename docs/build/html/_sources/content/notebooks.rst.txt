Notebooks
+++++++++

The notebooks are Colab examples grouped by topic. Each entry links to the related documentation page.


Getting Started
===============

* `01 Optimization Model <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/01%20Optimization%20Model.ipynb>`_: build an ``optModel`` from a GurobiPy / COPT / Pyomo / OR-Tools / MPAX backend. Pairs with :doc:`getting_started/model`.
* `02 Optimization Dataset <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/02%20Optimization%20Dataset.ipynb>`_: generate synthetic data and wrap it in ``optDataset``. Pairs with :doc:`getting_started/data`.
* `03 Training and Testing <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/03%20Training%20and%20Testing.ipynb>`_: train method families on a shortest-path dataset. Pairs with :doc:`getting_started/training` and :doc:`getting_started/evaluation`.


Method Deep Dives
=================

* `04 CaVE for Binary Linear Programs <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/04%20CaVE%20for%20Binary%20Linear%20Programs.ipynb>`_: train with the cone-aligned CaVE loss on TSP. Pairs with the *Cone-Aligned Estimation* section of :doc:`getting_started/function`.
* `08 kNN Robust Losses <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/08%20kNN%20Robust%20Losses.ipynb>`_: train with the kNN robust loss via ``optDatasetKNN``. Pairs with the *optDatasetKNN* section of :doc:`getting_started/data`.


GPU Acceleration
================

* `09 Solving on MPAX with PDHG <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/09%20Solving%20on%20MPAX%20with%20PDHG.ipynb>`_: batch-solve LPs on GPU via MPAX, end-to-end without CPU round-trips. See the MPAX backend (``optMpaxModel``) in the *Solver Backend Subclass* section of :doc:`getting_started/model`.
* `10 JAX Frontend <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/10%20JAX%20Frontend.ipynb>`_: train PyEPO losses in JAX/Flax with ``jax.grad``. MPAX is GPU-native and jittable; non-JAX backends run through ``jax.pure_callback``. Pairs with :doc:`frontends/jax`.


Applied Examples
================

* `05 2D Knapsack Solution Visualization <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/05%202D%20Knapsack%20Solution%20Visualization.ipynb>`_: visualize selected items for a 2D knapsack instance to inspect what the trained predictor is doing.
* `06 Warcraft Shortest Path <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/06%20Warcraft%20Shortest%20Path.ipynb>`_: train an image-based shortest-path predictor on the Warcraft terrain dataset, using a CNN encoder feeding into PyEPO.
* `07 Real-World Energy Scheduling <https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/07%20Real-World%20Energy%20Scheduling.ipynb>`_: apply PyEPO to an energy scheduling dataset with measured demand data.
