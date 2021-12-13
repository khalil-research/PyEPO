:mod:`spo.eval.metrics`
=======================

.. py:module:: spo.eval.metrics

.. autoapi-nested-parse::

   metrics for SKlearn model



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   spo.eval.metrics.SPOError
   spo.eval.metrics.makeSkScorer
   spo.eval.metrics.makeAutoSkScorer
   spo.eval.metrics.testMSE
   spo.eval.metrics.makeTestMSEScorer


.. function:: SPOError(pred_cost, true_cost, model_type, args)

   A function to calculate normalized true SPO

   :param pred_cost: predicted costs
   :type pred_cost: array
   :param true_cost: true costs
   :type true_cost: array
   :param model_type: optModel class type
   :type model_type: ABCMeta
   :param args: optModel args
   :type args: dict

   :returns: true SPO losses
   :rtype: float


.. function:: makeSkScorer(omodel)

   A function to create sklearn scorer

   :param omodel: optimization model
   :type omodel: optModel

   :returns: callable object that returns a scalar score; less is better.
   :rtype: scorer


.. function:: makeAutoSkScorer(omodel)

   A function to create Auto-SKlearn scorer

   :param omodel: optimization model
   :type omodel: optModel

   :returns: callable object that returns a scalar score; less is better.
   :rtype: scorer


.. function:: testMSE(pred_cost, true_cost, model_type, args)

   A function to calculate MSE for testing

   :param pred_cost: predicted costs
   :type pred_cost: array
   :param true_cost: true costs
   :type true_cost: array
   :param model_type: optModel class type
   :type model_type: ABCMeta
   :param args: optModel args
   :type args: dict

   :returns: mse
   :rtype: float


.. function:: makeTestMSEScorer(omodel)

   A function to create MSE scorer for testing

   :param omodel: optimization model
   :type omodel: optModel

   :returns: callable object that returns a scalar score; less is better.
   :rtype: scorer


