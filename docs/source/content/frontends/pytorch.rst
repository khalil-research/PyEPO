PyTorch Frontend
++++++++++++++++

The PyTorch frontend lives in ``pyepo.func``. Each training method wraps an ``optModel`` and can be used with PyTorch optimizers. Method selection, training-loop templates, and API details are in :doc:`../getting_started/function`; this page summarizes the calling conventions.


Basic Pattern
=============

Instantiate a method, call it inside the PyTorch training loop, and backpropagate through the returned loss:

.. code-block:: python

   spo = pyepo.func.SPOPlus(optmodel, processes=1)
   loss = spo(pred_cost, true_cost, true_sol, true_obj)
   loss.backward()

The ``optModel`` defines the feasible region. The prediction model produces ``pred_cost``; ``pyepo.func`` updates the optimization objective, solves the problem, and supplies the gradient rule used by PyTorch.


Batch Inputs
============

The usual batch format comes from ``optDataset``:

.. code-block:: python

   for feat, true_cost, true_sol, true_obj in dataloader:
       pred_cost = predmodel(feat)
       loss = spo(pred_cost, true_cost, true_sol, true_obj)

The tensors mean:

* ``feat``: input features for the prediction model.
* ``pred_cost``: predicted objective coefficients.
* ``true_cost``: ground-truth objective coefficients.
* ``true_sol``: optimal solution under ``true_cost``.
* ``true_obj``: optimal objective value under ``true_cost``.

Some methods use only a subset of these values; the per-method inputs are listed in the summary table of :doc:`../getting_started/function`.


Loss-returning and Solution-returning Methods
=============================================

PyEPO methods use two interfaces:

* **Loss-returning methods** return a scalar loss. Call ``loss.backward()`` directly. Examples: ``SPOPlus``, ``PFY``, ``NCE``, ``lsLTR``, ``CaVE``.
* **Solution-returning methods** return a predicted or perturbed solution. Define a task loss on top, then backpropagate through that loss. Examples: ``DPO``, ``DBB``, ``NID``, ``RFWO``, ``IMLE``.


Shared Options
==============

Common constructor options:

* ``processes`` controls the worker pool used for batch solving.
* ``solve_ratio`` enables solution-pool caching when set below ``1`` (CaVE repurposes this knob for its projection branch; see :doc:`../advanced/cave`).
* ``dataset`` seeds the solution pool whenever ``solve_ratio < 1``; contrastive and ranking methods require it always.
* ``reduction`` controls how per-instance losses are aggregated when the method supports it.
