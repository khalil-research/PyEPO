:mod:`spo.model.omo.knapsack`
=============================

.. py:module:: spo.model.omo.knapsack

.. autoapi-nested-parse::

   Knapsack problem



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   spo.model.omo.knapsack.knapsackModel
   spo.model.omo.knapsack.knapsackModelRel



.. py:class:: knapsackModel(weights, capacity, solver='glpk')

   Bases: :class:`spo.model.omo.optOmoModel`

   This class is optimization model for knapsack problem

   :param weights: weights of items
   :type weights: ndarray
   :param capacity: total capacity
   :type capacity: ndarray
   :param solver: optimization solver for pyomo
   :type solver: str

   .. method:: num_cost(self)
      :property:

      number of cost to be predicted


   .. method:: _getModel(self)

      A method to build pyomo model


   .. method:: relax(self)

      A method to relax model



.. py:class:: knapsackModelRel(weights, capacity, solver='glpk')

   Bases: :class:`spo.model.omo.knapsack.knapsackModel`

   This class is relaxed optimization model for knapsack problem.

   .. method:: _getModel(self)

      A method to build pyomo


   .. method:: relax(self)

      A forbidden method to relax MIP model



