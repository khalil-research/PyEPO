:mod:`spo.data.dataset`
=======================

.. py:module:: spo.data.dataset

.. autoapi-nested-parse::

   Torch Dataset for optimization



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   spo.data.dataset.optDataset



.. py:class:: optDataset(model, feats, costs)

   Bases: :class:`torch.utils.data.Dataset`

   This class is Torch Dataset for optimization problems.

   :param model: an instance of optModel
   :type model: optModel
   :param feats: data features
   :type feats: ndarray
   :param costs: costs of objective function
   :type costs: ndarray

   .. method:: _getSols(self)

      A method to get optimal solutions for all cost vectors


   .. method:: _solve(self, cost)

      A method to solve optimization problem to get an optimal solution with given cost

      :param cost: cost of objective function
      :type cost: ndarray

      :returns: optimal solution (ndarray) and objective value (float)
      :rtype: tuple


   .. method:: __len__(self)

      A method to get data size

      :returns: the number of optimization problems
      :rtype: int


   .. method:: __getitem__(self, index)

      A method to retrieve data

      :param index: data index
      :type index: int

      :returns: data features (tensor), costs (tensor), optimal solutions (tensor) and objective values (tensor)
      :rtype: tuple



