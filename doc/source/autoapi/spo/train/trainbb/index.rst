:mod:`spo.train.trainbb`
========================

.. py:module:: spo.train.trainbb

.. autoapi-nested-parse::

   Train with SPO+ loss



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   spo.train.trainbb.trainBB


.. function:: trainBB(reg, model, optimizer, trainloader, testloader=None, logdir='./logs', epoch=50, processes=1, bb_lambd=10, l1_lambd=0, l2_lambd=0, log=0)

   A function to train PyTorch nn with SPO+ loss

   :param reg: PyTorch neural network regressor
   :type reg: nn
   :param model: optimization model
   :type model: optModel
   :param optimizer: PyTorch optimizer
   :type optimizer: optim
   :param trainloader: PyTorch DataLoader for train set
   :type trainloader: DataLoader
   :param testloader: PyTorch DataLoader for test set
   :type testloader: DataLoader
   :param epoch: number of training epochs
   :type epoch: int
   :param processes: processes (int): number of processors, 1 for single-core, 0 for all of cores
   :param bb_lambd: Black-Box parameter for function smoothing
   :type bb_lambd: float
   :param l1_lambd: regularization weight of l1 norm
   :type l1_lambd: float
   :param l2_lambd: regularization weight of l2 norm
   :type l2_lambd: float
   :param log: step size of evlaution and log
   :type log: int


