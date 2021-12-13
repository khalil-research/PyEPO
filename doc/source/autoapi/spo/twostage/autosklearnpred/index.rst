:mod:`spo.twostage.autosklearnpred`
===================================

.. py:module:: spo.twostage.autosklearnpred

.. autoapi-nested-parse::

   Two-stage model with Scikit-learn predictor



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   spo.twostage.autosklearnpred.NoPreprocessing



Functions
~~~~~~~~~

.. autoapisummary::

   spo.twostage.autosklearnpred.autoSklearnPred


.. py:class:: NoPreprocessing(**kwargs)

   Bases: :class:`autosklearn.pipeline.components.base.AutoSklearnPreprocessingAlgorithm`

   This is class of NoPreprocessing component for auto-sklearn

   .. method:: fit(self, X, Y=None)


   .. method:: transform(self, X)


   .. method:: get_properties(dataset_properties=None)
      :staticmethod:


   .. method:: get_hyperparameter_search_space(dataset_properties=None)
      :staticmethod:



.. function:: autoSklearnPred(omodel, seed)

   Two-stage prediction and optimization with auto-sklearn.

   :param omodel: optimization model
   :type omodel: optModel

   :returns: Auto-SKlearn multi-output regression model
   :rtype: AutoSklearnRegressor


