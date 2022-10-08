.. PyEPO documentation master file, created by
   sphinx-quickstart on Mon Aug  9 14:15:50 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ./images/logo1.png
   :width: 1000

Welcome to PyEPO's documentation!
=================================
This is the documentation of ``PyEPO`` (PyTorch-based End-to-End Predict-then-Optimize Library Tool), which aims to provide end-to-end methods for predict-then-optimize tasks.


Sample Code
+++++++++++

.. code-block:: python

   import random

   import gurobipy as gp
   from gurobipy import GRB

   from pyepo.model.grb import optGrbModel

   class myModel(optGrbModel):

       def _getModel(self):
           # create a model
           m = gp.Model()
           # variables
           x = m.addVars(5, name="x", vtype=GRB.BINARY)
           # model sense
           m.modelSense = GRB.MAXIMIZE
           # constraints
           m.addConstr(3 * x[0] + 4 * x[1] + 3 * x[2] + 6 * x[3] + 4 * x[4] <= 12)
           m.addConstr(4 * x[0] + 5 * x[1] + 2 * x[2] + 3 * x[3] + 5 * x[4] <= 10)
           m.addConstr(5 * x[0] + 4 * x[1] + 6 * x[2] + 2 * x[3] + 3 * x[4] <= 15)
           return m, x

   # set optimization model
   optmodel = myModel()
   # init SPO+ loss
   spo = pyepo.func.SPOPlus(optmodel, processes=1)


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   content/intro
   content/install
   content/tutorial
   content/api
   content/ref


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
