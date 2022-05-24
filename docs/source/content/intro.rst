.. image:: ../images/logo1.png
  :width: 1000

Introduction
++++++++++++

``PyEPO`` (PyTorch-based End-to-End Predict-then-Optimize Tool) is a Python-based, open-source software that supports modeling and solving predict-then-optimize problems with linear objective function.

The core capability of ``PyEPO`` is to build optimization models with `GurobiPy <https://www.gurobi.com/>`_, `Pyomo <http://www.pyomo.org/>`_, or any other solvers and algorithms, then embed the optimization model into an artificial neural network for the end-to-end training. For this purpose, ``PyEPO`` implements SPO+ loss and differentiable Black-Box optimizer as `PyTorch <https://pytorch.org/>`_ autograd functions.

End-to-End Predict-then-Optimize Framework
-----------------------------------------

A labeled dataset :math:`\mathcal{D}` of :math:`(x,c)` pairs is used to fit a machine learning model that directly minimizes decision error. The critical component is an optimization solver which is embedded into a neural network.

.. image:: ../images/e2e.png
   :width: 900
