:mod:`spo.func.spoplus`
=======================

.. py:module:: spo.func.spoplus

.. autoapi-nested-parse::

   SPO+ Loss function



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   spo.func.spoplus.SPOPlus



Functions
~~~~~~~~~

.. autoapisummary::

   spo.func.spoplus.solveWithObj4Par


.. function:: solveWithObj4Par(cost, args, model_type)

   A global function to solve function in parallel processors

   :param cost: cost of objective function
   :type cost: ndarray
   :param args: optModel args
   :type args: dict
   :param model_type: optModel class type
   :type model_type: ABCMeta

   :returns: optimal solution (list) and objective value (float)
   :rtype: tuple


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



