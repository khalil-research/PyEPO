:mod:`spo.model.grb.grbmodel`
=============================

.. py:module:: spo.model.grb.grbmodel

.. autoapi-nested-parse::

   Abstract optimization model based on gurobipy



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   spo.model.grb.grbmodel.optGRBModel



.. py:class:: optGRBModel

   Bases: :class:`spo.model.optModel`

   This is an abstract class for Gurobi-based optimization model

   .. method:: __repr__(self)

      Return repr(self).


   .. method:: setObj(self, c)

      A method to set objective function

      :param c: cost of objective function
      :type c: ndarray


   .. method:: solve(self)

      A method to solve model

      :returns: optimal solution (list) and objective value (float)
      :rtype: tuple


   .. method:: copy(self)

      A method to copy model

      :returns: new copied model
      :rtype: optModel


   .. method:: addConstr(self, coefs, rhs)

      A method to add new constraint

      :param coefs: coeffcients of new constraint
      :type coefs: ndarray
      :param rhs: right-hand side of new constraint
      :type rhs: float

      :returns: new model with the added constraint
      :rtype: optModel



