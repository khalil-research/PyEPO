:mod:`spo.func`
===============

.. py:module:: spo.func

.. autoapi-nested-parse::

   Pytorch autograd function for SPO training



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   blackbox/index.rst
   spoplus/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   spo.func.blackboxOpt
   spo.func.SPOPlus



.. py:class:: blackboxOpt(model, lambd=10, processes=1)

   Bases: :class:`torch.autograd.Function`

   block-box optimizer function, which is diffenretiable to introduce blocks
   into neural networks.

   For block-box, the objective function is linear and constraints are known
   and fixed, but the cost vector need to be predicted from contextual data.

   The block-box approximate gradient of optimizer smoothly. Thus, allows us to
   design an algorithm based on stochastic gradient descent.

   :param model: optimization model
   :type model: optModel
   :param lambd: Black-Box parameter for function smoothing
   :type lambd: float
   :param processes: number of processors, 1 for single-core, 0 for all of cores
   :type processes: int

   .. method:: forward(ctx, pred_cost)
      :staticmethod:

      Forward pass in neural network.

      :param pred_cost: predicted costs

      :returns: predicted solutions
      :rtype: tensor


   .. method:: backward(ctx, grad_output)
      :staticmethod:

      Backward pass in neural network



.. py:class:: SPOPlus(model, processes=1)

   Bases: :class:`torch.autograd.Function`

   SPO+ Loss function, a surrogate loss function of SPO Loss, which measures
   the decision error (optimality gap) of optimization problem.

   For SPO/SPO+ Loss, the objective function is linear and constraints are
   known and fixed, but the cost vector need to be predicted from contextual
   data.

   The SPO+ Loss is convex with subgradient. Thus, allows us to design an
   algorithm based on stochastic gradient descent.

   :param model: optimization model
   :type model: optModel
   :param processes: number of processors, 1 for single-core, 0 for all of cores
   :type processes: int

   .. method:: forward(ctx, pred_cost, true_cost, true_sol, true_obj)
      :staticmethod:

      Forward pass in neural network

      :param pred_cost: predicted costs
      :type pred_cost: tensor
      :param true_cost: true costs
      :type true_cost: tensor
      :param true_sol: true solutions
      :type true_sol: tensor
      :param true_obj: true objective values
      :type true_obj: tensor

      :returns: SPO+ loss
      :rtype: tensor


   .. method:: backward(ctx, grad_output)
      :staticmethod:

      Backward pass in neural network



