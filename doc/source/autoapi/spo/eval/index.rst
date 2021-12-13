:mod:`spo.eval`
===============

.. py:module:: spo.eval

.. autoapi-nested-parse::

   Performance evaluation



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   metrics/index.rst
   truespo/index.rst
   unambspo/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   spo.eval.calTrueSPO
   spo.eval.trueSPO
   spo.eval.calUnambSPO
   spo.eval.unambSPO
   spo.eval.SPOError
   spo.eval.makeSkScorer
   spo.eval.makeAutoSkScorer


.. function:: calTrueSPO(omodel, pred_cost, true_cost, true_obj)

   A function to calculate normalized true SPO for a batch

   :param omodel: optimization model
   :type omodel: optModel
   :param pred_cost: predicted costs
   :type pred_cost: tensor
   :param true_cost: true costs
   :type true_cost: tensor
   :param true_obj: true optimal objective values
   :type true_obj: tensor

   :returns: true SPO losses
   :rtype: float


.. function:: trueSPO(pmodel, omodel, dataloader)

   A function to evaluate model performence with normalized true SPO

   :param pmodel: neural network predictor
   :type pmodel: nn
   :param omodel: optimization model
   :type omodel: optModel
   :param dataloader: Torch dataloader from optDataSet
   :type dataloader: DataLoader

   :returns: true SPO loss
   :rtype: float


.. function:: calUnambSPO(omodel, pred_cost, true_cost, true_obj, tolerance=1e-05)

   A function to calculate normalized unambiguous SPO for a batch

   :param omodel: optimization model
   :type omodel: optModel
   :param pred_cost: predicted costs
   :type pred_cost: tensor
   :param true_cost: true costs
   :type true_cost: tensor
   :param true_obj: true optimal objective values
   :type true_obj: tensor

   :returns: unambiguous SPO losses
   :rtype: float


.. function:: unambSPO(pmodel, omodel, dataloader, tolerance=1e-05)

   A function to evaluate model performence with normalized unambiguous SPO

   :param pmodel: neural network predictor
   :type pmodel: nn
   :param omodel: optimization model
   :type omodel: optModel
   :param dataloader: Torch dataloader from optDataSet
   :type dataloader: DataLoader

   :returns: unambiguous SPO loss
   :rtype: float


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


