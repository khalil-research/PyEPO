.. image:: ../images/logo1.png
  :width: 1000

Introduction
++++++++++++

``PyEPO`` (PyTorch-based End-to-End Predict-then-Optimize Tool) is a Python library for modeling and solving predict-then-optimize problems with linear objective functions.

``PyEPO`` builds optimization models with `GurobiPy <https://www.gurobi.com/>`_, `COPT <https://shanshu.ai/copt>`_, `Pyomo <http://www.pyomo.org/>`_, `Google OR-Tools <https://developers.google.com/optimization>`_, `MPAX <https://github.com/MIT-Lu-Lab/MPAX>`_, or any custom solver, and embeds them into neural networks for end-to-end training. All decision-focused learning methods are implemented as `PyTorch <https://pytorch.org/>`_ autograd modules, grouped into the following families:

* **Surrogate losses** -- SPO+, perturbation gradient (PG)
* **Perturbed methods** -- differentiable perturbed optimizers (DPO), perturbed Fenchel-Young loss (PFYL), I-MLE / AI-MLE
* **Regularized methods** -- L2-regularized Frank-Wolfe (RFWO), regularized Frank-Wolfe with Fenchel-Young loss (RFYL)
* **Black-box methods** -- differentiable black-box optimizer (DBB), negative identity backpropagation (NID)
* **Cone-aligned estimation** -- CaVE (binary linear programs only)
* **Contrastive methods** -- noise contrastive estimation (NCE), contrastive MAP (CMAP)
* **Learning to rank** -- pointwise, pairwise, listwise LTR

End-to-End Predict-then-Optimize Framework
------------------------------------------

Given a labeled dataset :math:`\mathcal{D}` of feature-cost pairs :math:`(\mathbf{x}, \mathbf{c})` or feature-solution pairs :math:`(\mathbf{x}, \mathbf{w})`, a neural network is trained to directly minimize the decision error, rather than the prediction error of cost coefficients.

.. image:: ../images/e2e.png
   :width: 900
