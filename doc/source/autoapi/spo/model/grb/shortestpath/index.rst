:mod:`spo.model.grb.shortestpath`
=================================

.. py:module:: spo.model.grb.shortestpath

.. autoapi-nested-parse::

   Shortest path problem



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   spo.model.grb.shortestpath.shortestPathModel



.. py:class:: shortestPathModel(grid)

   Bases: :class:`spo.model.grb.optGRBModel`

   This class is optimization model for shortest path problem

   :param grid: size of grid network

   .. method:: _getArcs(self)

      A method to get list of arcs for grid network

      :returns: arcs
      :rtype: list


   .. method:: num_cost(self)
      :property:


   .. method:: _getModel(self)

      A method to build Gurobi model



