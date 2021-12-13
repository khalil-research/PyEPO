:mod:`spo.twostage.sklearnpred`
===============================

.. py:module:: spo.twostage.sklearnpred

.. autoapi-nested-parse::

   Two-stage model with Scikit-learn predictor



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   spo.twostage.sklearnpred.sklearnPred


.. function:: sklearnPred(pmodel)

   Two-stage prediction and optimization with scikit-learn.

   :param pmodel: scikit-learn regression model
   :type pmodel: Regressor

   :returns: scikit-learn multi-output regression model
   :rtype: MultiOutputRegressor


