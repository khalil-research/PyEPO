.. image:: ../images/logo1.png
  :width: 1000

Introduction
++++++++++++

``PyEPO`` (PyTorch-based End-to-End Predict-then-Optimize Tool) is a Python-based, open-source software that supports modeling and solving predict-then-optimize problems with the linear objective function.

The core capability of ``PyEPO`` is to build optimization models with `GurobiPy <https://www.gurobi.com/>`_, `Pyomo <http://www.pyomo.org/>`_, or any other solvers and algorithms, then embed the optimization model into an artificial neural network for the end-to-end training. For this purpose, ``PyEPO`` implements SPO+ loss, differentiable black-box optimizer, differentiable perturbed optimizers, and Fenchel-Young loss with Perturbation as `PyTorch <https://pytorch.org/>`_ autograd modules.

End-to-End Predict-then-Optimize Framework
-----------------------------------------

A labeled dataset :math:`\mathcal{D}` of feature-cost pairs  :math:`(x,c)` or feature-solution pairs :math:`(x,w)` is used to fit a machine learning model (especially a deep neural network) that directly minimizes the decision error. The critical component is a differentiable optimization solver which is embedded into a neural network.

.. image:: ../images/e2e.png
   :width: 900
