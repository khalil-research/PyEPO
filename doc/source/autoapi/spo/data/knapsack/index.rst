:mod:`spo.data.knapsack`
========================

.. py:module:: spo.data.knapsack

.. autoapi-nested-parse::

   Knapsack problem



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   spo.data.knapsack.genData


.. function:: genData(num_data, num_features, num_items, dim=1, deg=1, noise_width=0, seed=135)

   A function to generate synthetic data and features for knapsack

   :param num_data: number of data points
   :type num_data: int
   :param num_features: dimension of features
   :type num_features: int
   :param num_items: number of items
   :type num_items: int
   :param deg: data polynomial degree
   :type deg: int
   :param dim: dimension of multi-dimensional knapsack
   :type dim: int
   :param noise_withd: half witdth of data random noise
   :type noise_withd: float
   :param seed: random seed
   :type seed: int

   :returns: weights of items (ndarray), data features (ndarray), costs (ndarray)
   :rtype: tuple


