:mod:`spo.model.grb.tsp`
========================

.. py:module:: spo.model.grb.tsp

.. autoapi-nested-parse::

   Traveling salesman probelm



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   spo.model.grb.tsp.tspABModel
   spo.model.grb.tsp.tspGGModel
   spo.model.grb.tsp.tspGGModelRel
   spo.model.grb.tsp.tspDFJModel
   spo.model.grb.tsp.tspMTZModel
   spo.model.grb.tsp.tspMTZModelRel



.. py:class:: tspABModel(num_nodes)

   Bases: :class:`spo.model.grb.optGRBModel`

   This class is optimization model for traveling salesman problem.
   This model is for further implementation of different formulation.

   :param num_nodes: number of nodes

   .. method:: num_cost(self)
      :property:


   .. method:: copy(self)

      A method to copy model

      :returns: new copied model
      :rtype: optModel


   .. method:: getTour(self, sol)

      A method to get a tour from solution

      :param sol: solution
      :type sol: list

      :returns: a TSP tour
      :rtype: list



.. py:class:: tspGGModel(num_nodes)

   Bases: :class:`spo.model.grb.tsp.tspABModel`

   This class is optimization model for traveling salesman problem.
   This model is based on Gavish–Graves (GG) formulation.

   :param num_nodes: number of nodes

   .. method:: _getModel(self)

      A method to build Gurobi model


   .. method:: setObj(self, c)

      set objective function


   .. method:: solve(self)

      solve model


   .. method:: addConstr(self, coefs, rhs)

      A method to add new constraint

      :param coefs: coeffcients of new constraint
      :type coefs: ndarray
      :param rhs: right-hand side of new constraint
      :type rhs: float

      :returns: new model with the added constraint
      :rtype: optModel


   .. method:: relax(self)

      A method to relax model



.. py:class:: tspGGModelRel(num_nodes)

   Bases: :class:`spo.model.grb.tsp.tspGGModel`

   This class is relaxed optimization model for Gavish–Graves (GG) formulation.

   .. method:: _getModel(self)

      A method to build Gurobi model


   .. method:: solve(self)

      A method to solve model

      :returns: optimal solution (list) and objective value (float)
      :rtype: tuple


   .. method:: relax(self)

      A forbidden method to relax MIP model


   .. method:: getTour(self, sol)

      A forbidden method to get a tour from solution



.. py:class:: tspDFJModel(num_nodes)

   Bases: :class:`spo.model.grb.tsp.tspABModel`

   This class is optimization model for traveling salesman problem.
   This model is based on Danzig–Fulkerson–Johnson (DFJ) formulation and
   constraint generation.

   :param num_nodes: number of nodes

   .. method:: _getModel(self)

      A method to build Gurobi model


   .. method:: _subtourelim(model, where)
      :staticmethod:

      A static method to add lazy constraints for subtour elimination


   .. method:: setObj(self, c)

      set objective function


   .. method:: solve(self)

      solve model


   .. method:: addConstr(self, coefs, rhs)

      A method to add new constraint

      :param coefs: coeffcients of new constraint
      :type coefs: ndarray
      :param rhs: right-hand side of new constraint
      :type rhs: float

      :returns: new model with the added constraint
      :rtype: optModel



.. py:class:: tspMTZModel(num_nodes)

   Bases: :class:`spo.model.grb.tsp.tspABModel`

   This class is optimization model for traveling salesman problem.
   This model is based on Miller-Tucker-Zemlin (MTZ) formulation.

   :param num_nodes: number of nodes

   .. method:: _getModel(self)

      A method to build Gurobi model


   .. method:: setObj(self, c)

      set objective function


   .. method:: solve(self)

      solve model


   .. method:: addConstr(self, coefs, rhs)

      A method to add new constraint

      :param coefs: coeffcients of new constraint
      :type coefs: ndarray
      :param rhs: right-hand side of new constraint
      :type rhs: float

      :returns: new model with the added constraint
      :rtype: optModel


   .. method:: relax(self)

      A method to relax model



.. py:class:: tspMTZModelRel(num_nodes)

   Bases: :class:`spo.model.grb.tsp.tspMTZModel`

   This class is relaxed optimization model for Miller-Tucker-Zemlin (MTZ)
   formulation.

   .. method:: _getModel(self)

      A method to build Gurobi model


   .. method:: solve(self)

      A method to solve model

      :returns: optimal solution (list) and objective value (float)
      :rtype: tuple


   .. method:: relax(self)

      A forbidden method to relax MIP model


   .. method:: getTour(self, sol)

      A forbidden method to get a tour from solution



