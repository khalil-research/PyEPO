:mod:`spo.model.omo`
====================

.. py:module:: spo.model.omo

.. autoapi-nested-parse::

   Optimization Model based on pyomo



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   knapsack/index.rst
   omomodel/index.rst
   shortestpath/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   spo.model.omo.optOmoModel
   spo.model.omo.shortestPathModel



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



.. py:class:: shortestPathModel(grid, solver='glpk')

   Bases: :class:`spo.model.omo.optOmoModel`

   This class is optimization model for shortest path problem

   :param grid: size of grid network
   :param solver: optimization solver

   .. method:: _getArcs(self)

      A method to get list of arcs for grid network

      :returns: arcs
      :rtype: list


   .. method:: num_cost(self)
      :property:

      number of cost to be predicted


   .. method:: _getModel(self)

      A method to build pyomo model



