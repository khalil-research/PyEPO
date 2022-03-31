# PyEPO: A PyTorch-based End-to-End Predict-then-Optimize Library for Linear and Integer Programming

``PyEPO`` (PyTorch-based End-to-End Predict-then-Optimize Tool) is a Python-based, open-source software package that supports modeling and solving predict-then-optimize problems with the linear objective function. The core capability of ``PyEPO`` is to build your optimization models with [GurobiPy](https://www.gurobi.com/), [Pyomo](http://www.pyomo.org/), or any other solvers and algorithms, then embed the optimization model into an artificial neural network for the end-to-end training. For this purpose, ``PyEPO`` implemented SPO+ loss and differentiable Black-Box optimizer as [PyTorch](https://pytorch.org/) autograd functions.

## Dependencies

* [SciPy](https://scipy.org/)
* [Pathos](https://pathos.readthedocs.io/)
* [Pyomo](http://www.pyomo.org/)
* [Gurobi](https://www.gurobi.com/)
* [Scikit Learn](https://scikit-learn.org/)
* [PyTorch](http://pytorch.org/)

## Sample Code

```python
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pyepo
from pyepo.model.grb import optGrbModel
import torch
from torch import nn
from torch.utils.data import DataLoader

# generate data
num_data = 1000 # number of data
num_feat = 5 # size of feature
num_item = 10 # number of items
weights, x, c = pyepo.data.knapsack.genData(num_data, num_feat, num_item, dim=3, deg=4, noise_width=0.5, seed=135)

# optimization model
class myModel(optGrbModel):
    def __init__(self, weights):
        self.weights = np.array(weights)
        self.num_item = len(weights[0])
        super().__init__()
        
    def _getModel(self):
        # ceate a model
        m = gp.Model()
        # varibles
        x = m.addVars(self.num_item, name="x", vtype=GRB.BINARY)
        # sense (must be minimize)
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(gp.quicksum([self.weights[0,i] * x[i] for i in range(self.num_item)]) <= 7)
        m.addConstr(gp.quicksum([self.weights[1,i] * x[i] for i in range(self.num_item)]) <= 8)
        m.addConstr(gp.quicksum([self.weights[2,i] * x[i] for i in range(self.num_item)]) <= 9)
        return m, x
        
optmodel = myModel(weights)

# build dataset
dataset = pyepo.data.dataset.optDataset(optmodel, x, c)

# get data loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# build linear model
class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_feat, num_item)

    def forward(self, x):
        out = self.linear(x)
        return out
# init
predmodel = LinearRegression()
# set optimizer
optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-2)
# init SPO+ loss
spo = pyepo.func.SPOPlus(optmodel, processes=4)

# training
num_epochs = 100
for epoch in range(num_epochs):
    for data in dataloader:
        x, c, w, z = data
        # forward pass
        cp = predmodel(x)
        loss = spo.apply(cp, c, w, z).mean()
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
