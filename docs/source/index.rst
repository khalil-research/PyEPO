.. PyEPO documentation master file, created by
   sphinx-quickstart on Mon Aug  9 14:15:50 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ./images/logo1.png
   :width: 1000

Welcome to PyEPO's documentation!
=================================
This is the documentation of ``PyEPO`` (PyTorch-based End-to-End Predict-then-Optimize Tool), which aims to provide end-to-end methods for predict-then-optimize tasks.


Quick Example
+++++++++++++

End-to-end training of a shortest-path predictor on a 5x5 grid with the SPO+ loss:

.. code-block:: python

   import pyepo
   import torch
   from torch import nn
   from torch.utils.data import DataLoader

   # optimization model: 5x5 grid shortest path
   grid = (5, 5)
   optmodel = pyepo.model.shortestPathModel(grid)

   # synthetic data and dataset
   x, c = pyepo.data.shortestpath.genData(
       num_data=1000, num_features=5, grid=grid, deg=4, noise_width=0.5, seed=135,
   )
   dataset = pyepo.data.dataset.optDataset(optmodel, x, c)
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

   # linear predictor and SPO+ loss
   predmodel = nn.Linear(5, 40)
   spo = pyepo.func.SPOPlus(optmodel, processes=1)
   optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-3)

   # end-to-end training
   for epoch in range(10):
       for xb, cb, wb, zb in dataloader:
           loss = spo(predmodel(xb), cb, wb, zb)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

   # decision quality (on the training set here; split off a test set for real evaluation)
   print("Training regret:", pyepo.metric.regret(predmodel, optmodel, dataloader))


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   content/intro
   content/install
   content/tutorial
   content/ref


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
