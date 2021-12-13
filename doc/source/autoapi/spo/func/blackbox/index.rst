:mod:`spo.func.blackbox`
========================

.. py:module:: spo.func.blackbox

.. autoapi-nested-parse::

   Diffenretiable Black-box optimization function



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   spo.func.blackbox.blackboxOpt



Functions
~~~~~~~~~

.. autoapisummary::

   spo.func.blackbox.solveWithObj4Par


.. function:: solveWithObj4Par(cost, args, model_type)

   A global function to solve function in parallel processors

   :param cost: cost of objective function
   :type cost: ndarray
   :param args: optModel args
   :type args: dict
   :param model_type: optModel class type
   :type model_type: ABCMeta

   :returns: optimal solution
   :rtype: list


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



