:mod:`spo.eval.unambspo`
========================

.. py:module:: spo.eval.unambspo

.. autoapi-nested-parse::

   Unambiguous SPO loss



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   spo.eval.unambspo.unambSPO
   spo.eval.unambspo.calUnambSPO


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


