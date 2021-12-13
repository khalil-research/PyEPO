:mod:`spo.data.tsp`
===================

.. py:module:: spo.data.tsp

.. autoapi-nested-parse::

   Traveling salesman probelm



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   spo.data.tsp.genData


.. function:: genData(num_data, num_features, num_nodes, deg=1, noise_width=0, seed=135)

   A function to generate synthetic data and features for travelling salesman

   :param num_data: number of data points
   :type num_data: int
   :param num_features: dimension of features
   :type num_features: int
   :param num_nodes: number of nodes
   :type num_nodes: int
   :param deg: data polynomial degree
   :type deg: int
   :param noise_withd: half witdth of data random noise
   :type noise_withd: float
   :param seed: random seed
   :type seed: int

   :returns: data features (ndarray), costs (ndarray)
   :rtype: tuple


