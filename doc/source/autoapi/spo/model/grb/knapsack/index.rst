:mod:`spo.model.grb.knapsack`
=============================

.. py:module:: spo.model.grb.knapsack

.. autoapi-nested-parse::

   Knapsack problem



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   spo.model.grb.knapsack.knapsackModel
   spo.model.grb.knapsack.knapsackModelRel



.. py:class:: knapsackModel(weights, capacity)

   Bases: :class:`spo.model.grb.optGRBModel`

   This class is optimization model for knapsack problem

   :param weights: weights of items
   :type weights: ndarray
   :param capacity: total capacity
   :type capacity: ndarray

   .. method:: num_cost(self)
      :property:


   .. method:: _getModel(self)

      A method to build Gurobi model


   .. method:: relax(self)

      A method to relax model



.. py:class:: knapsackModelRel(weights, capacity)

   Bases: :class:`spo.model.grb.knapsack.knapsackModel`

   This class is relaxed optimization model for knapsack problem.

   .. method:: _getModel(self)

      A method to build Gurobi


   .. method:: relax(self)

      A forbidden method to relax MIP model



