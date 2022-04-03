.. image:: ../images/logo1.png
  :width: 1000

Introduction
++++++++++++

``PyEPO`` (PyTorch-based End-to-End Predict-and-Optimize Tool) is a Python-based, open-source software that supports modeling and solving predict-and-optimize problems with linear objective function.

The core capability of ``PyEPO`` is to build optimization models with `GurobiPy <https://www.gurobi.com/>`_, `Pyomo <http://www.pyomo.org/>`_, or any other solvers and algorithms, then embed the optimization model into an artificial neural network for the end-to-end training. For this purpose, ``PyEPO`` implements SPO+ loss and differentiable Black-Box optimizer as `PyTorch <https://pytorch.org/>`_ autograd functions.
