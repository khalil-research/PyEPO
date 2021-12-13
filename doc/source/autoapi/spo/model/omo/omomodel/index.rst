:mod:`spo.model.omo.omomodel`
=============================

.. py:module:: spo.model.omo.omomodel

.. autoapi-nested-parse::

   Abstract optimization model based on pyomo



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   spo.model.omo.omomodel.optOmoModel



.. py:class:: optOmoModel(solver='glpk')

   Bases: :class:`spo.model.optModel`

   This is an abstract class for pyomo-based optimization model

   :param solver: optimization solver

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



