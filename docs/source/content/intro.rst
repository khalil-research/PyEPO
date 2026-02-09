.. image:: ../images/logo1.png
  :width: 1000

Introduction
++++++++++++

``PyEPO`` (PyTorch-based End-to-End Predict-then-Optimize Tool) is a Python library for modeling and solving predict-then-optimize problems with linear objective functions.

``PyEPO`` allows users to build optimization models with `GurobiPy <https://www.gurobi.com/>`_, `Pyomo <http://www.pyomo.org/>`_, or any custom solver, and embed them into neural networks for end-to-end training. It provides a collection of decision-focused learning methods -- including SPO+ loss, differentiable black-box optimizers, perturbed optimizers, Fenchel-Young loss, contrastive losses, and learning-to-rank losses -- implemented as `PyTorch <https://pytorch.org/>`_ autograd modules.

End-to-End Predict-then-Optimize Framework
------------------------------------------

Given a labeled dataset :math:`\mathcal{D}` of feature-cost pairs :math:`(x,c)` or feature-solution pairs :math:`(x,w)`, a neural network is trained to directly minimize the decision error, rather than the prediction error of cost coefficients.

.. image:: ../images/e2e.png
   :width: 900
