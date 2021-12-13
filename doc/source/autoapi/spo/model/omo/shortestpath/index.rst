:mod:`spo.model.omo.shortestpath`
=================================

.. py:module:: spo.model.omo.shortestpath

.. autoapi-nested-parse::

   Shortest path problem



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   spo.model.omo.shortestpath.shortestPathModel



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



