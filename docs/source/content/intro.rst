.. image:: ../images/logo1.png
  :width: 1000

Introduction
++++++++++++

``PyEPO`` (PyTorch-based End-to-End Predict-then-Optimize Tool) is a Python library for modeling and solving predict-then-optimize problems with linear objective functions.

``PyEPO`` builds optimization models with `GurobiPy <https://www.gurobi.com/>`_, `COPT <https://shanshu.ai/copt>`_, `Pyomo <http://www.pyomo.org/>`_, `Google OR-Tools <https://developers.google.com/optimization>`_, `MPAX <https://github.com/MIT-Lu-Lab/MPAX>`_, or any custom solver or algorithm, and embeds them into neural networks for end-to-end training. All decision-focused learning methods are implemented as `PyTorch <https://pytorch.org/>`_ autograd modules, grouped into the following families:

* **Surrogate losses** -- SPO+, perturbation gradient (PG)
* **Perturbed methods** -- differentiable perturbed optimizers (DPO), perturbed Fenchel-Young loss (PFYL), I-MLE / AI-MLE
* **Regularized methods** -- L2-regularized Frank-Wolfe (RFWO), regularized Frank-Wolfe with Fenchel-Young loss (RFYL)
* **Black-box methods** -- differentiable black-box optimizer (DBB), negative identity backpropagation (NID)
* **Cone-aligned estimation** -- CaVE (binary linear programs only)
* **Contrastive methods** -- noise contrastive estimation (NCE), contrastive MAP (CMAP)
* **Learning to rank** -- pointwise, pairwise, listwise LTR

For end-to-end learning on **binary linear programs** (TSP, CVRP, knapsack, shortest path with binary edges), ``PyEPO`` ships **CaVE**, a first-order, GPU-native cone-projection solver written in pure PyTorch. Each backward pass becomes a handful of accelerated APGD iterations instead of a combinatorial solve -- typically training an order of magnitude faster than SPO+ on TSP-scale instances.

``PyEPO`` also integrates `MPAX <https://github.com/MIT-Lu-Lab/MPAX>`_, a JAX-based mathematical programming solver using the PDHG (Primal-Dual Hybrid Gradient) algorithm for GPU-accelerated optimization. MPAX brings three key advantages for end-to-end training: (1) **GPU-native solving** -- the first-order PDHG method is inherently parallelizable and runs efficiently on GPU; (2) **batch solving** -- an entire mini-batch of optimization instances is solved simultaneously via vectorization; (3) **no GPU-CPU data transfer overhead** -- both the neural network and the solver stay on GPU, eliminating the data-transfer bottleneck of CPU-side solvers like Gurobi.

End-to-End Predict-then-Optimize Framework
------------------------------------------

Given a labeled dataset :math:`\mathcal{D}` of feature-cost pairs :math:`(\mathbf{x}, \mathbf{c})` or feature-solution pairs :math:`(\mathbf{x}, \mathbf{w})`, a neural network is trained to directly minimize the decision error, rather than the prediction error of cost coefficients.

.. image:: ../images/e2e.png
   :width: 900

Publication
-----------

``PyEPO`` is the official implementation of the paper `PyEPO: A PyTorch-based End-to-End Predict-then-Optimize Library for Linear and Integer Programming <https://link.springer.com/article/10.1007/s12532-024-00255-x>`_ (Mathematical Programming Computation, 2024).

Citation
--------

If you use ``PyEPO`` in your research, please cite:

.. code-block:: bibtex

   @article{tang2024,
     title={PyEPO: a PyTorch-based end-to-end predict-then-optimize library for linear and integer programming},
     author={Tang, Bo and Khalil, Elias B},
     journal={Mathematical Programming Computation},
     issn={1867-2957},
     doi={10.1007/s12532-024-00255-x},
     year={2024},
     month={July},
     publisher={Springer}
   }

   @inproceedings{tang2024cave,
     title={CaVE: A Cone-Aligned Approach for Fast Predict-then-Optimize with Binary Linear Programs},
     author={Tang, Bo and Khalil, Elias B},
     booktitle={Integration of Constraint Programming, Artificial Intelligence, and Operations Research},
     pages={193--210},
     year={2024},
     publisher={Springer}
   }
