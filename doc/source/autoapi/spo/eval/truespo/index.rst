:mod:`spo.eval.truespo`
=======================

.. py:module:: spo.eval.truespo

.. autoapi-nested-parse::

   True SPO loss



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   spo.eval.truespo.trueSPO
   spo.eval.truespo.calTrueSPO


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


