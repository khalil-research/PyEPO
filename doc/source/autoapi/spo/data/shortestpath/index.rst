:mod:`spo.data.shortestpath`
============================

.. py:module:: spo.data.shortestpath

.. autoapi-nested-parse::

   Shortest path problem



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   spo.data.shortestpath.genData


.. function:: genData(num_data, num_features, grid, deg=1, noise_width=0, seed=135)

   A function to generate synthetic data and features for shortest path

   :param num_data: number of data points
   :type num_data: int
   :param num_features: dimension of features
   :type num_features: int
   :param grid: size of grid network
   :type grid: int, int
   :param deg: data polynomial degree
   :type deg: int
   :param noise_withd: half witdth of data random noise
   :type noise_withd: float
   :param seed: random seed
   :type seed: int

   :returns: data features (ndarray), costs (ndarray)
   :rtype: tuple


