:mod:`spo.model`
================

.. py:module:: spo.model

.. autoapi-nested-parse::

   Optimization Model based on solvers



Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   grb/index.rst
   omo/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   optmodel/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   spo.model.optModel



.. py:class:: optModel

   Bases: :class:`abc.ABC`

   This is an abstract class for optimization model

   .. method:: __repr__(self)

      Return repr(self).


   .. method:: num_cost(self)
      :property:

      number of cost to be predicted


   .. method:: _getModel(self)
      :abstractmethod:

      An abstract method to build a model from a optimization solver


   .. method:: setObj(self, c)
      :abstractmethod:

      An abstract method to set objective function

      :param c: cost of objective function
      :type c: ndarray


   .. method:: solve(self)
      :abstractmethod:

      An abstract method to solve model

      :returns: optimal solution (list) and objective value (float)
      :rtype: tuple


   .. method:: copy(self)

      An abstract method to copy model

      :returns: new copied model
      :rtype: optModel


   .. method:: addConstr(self, coefs, rhs)
      :abstractmethod:

      An abstract method to add new constraint

      :param coefs: coeffcients of new constraint
      :type coefs: ndarray
      :param rhs: right-hand side of new constraint
      :type rhs: float

      :returns: new model with the added constraint
      :rtype: optModel


   .. method:: relax(self)

      A unimplemented method to relax MIP model



