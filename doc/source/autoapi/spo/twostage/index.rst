:mod:`spo.twostage`
===================

.. py:module:: spo.twostage

.. autoapi-nested-parse::

   Two-stage predict then optimize model



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   autosklearnpred/index.rst
   sklearnpred/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   spo.twostage.sklearnPred


.. function:: sklearnPred(pmodel)

   Two-stage prediction and optimization with scikit-learn.

   :param pmodel: scikit-learn regression model
   :type pmodel: Regressor

   :returns: scikit-learn multi-output regression model
   :rtype: MultiOutputRegressor


