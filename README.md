# PyEPO: A PyTorch-based End-to-End Predict-and-Optimize Tool

<p align="center"><img width="100%" src="images/logo1.png" /></p>

## Publication

This repository is the official implementation of the paper:
[PyEPO: A PyTorch-based End-to-End Predict-then-Optimize Library for Linear and Integer Programming](http://www.optimization-online.org/DB_HTML/2022/06/8949.html)

Citation:
```
@article{tang2022pyepo,
  title={PyEPO: A PyTorch-based End-to-End Predict-then-Optimize Library for Linear and Integer Programming},
  author={Tang, Bo and Khalil, Elias B},
  journal={arXiv preprint arXiv:2206.14234},
  year={2022}
}
```


## Experiments

To reproduce the experiments in original paper, please use the code and follow the instruction in this [branch](https://github.com/khalil-research/PyEPO/tree/mpc).


## Introduction

``PyEPO`` (PyTorch-based End-to-End Predict-and-Optimize Tool) is a Python-based, open-source software that supports modeling and solving predict-and-optimize problems with the linear objective function. The core capability of ``PyEPO`` is to build optimization models with [GurobiPy](https://www.gurobi.com/), [Pyomo](http://www.pyomo.org/), or any other solvers and algorithms, then embed the optimization model into an artificial neural network for the end-to-end training. For this purpose, ``PyEPO`` implements SPO+ loss [[1]](https://doi.org/10.1287/mnsc.2020.3922) and differentiable Black-Box optimizer [[3]](https://arxiv.org/abs/1912.02175) as [PyTorch](https://pytorch.org/) autograd functions.


## Documentation

The official ``PyEPO`` docs can be found at [https://khalil-research.github.io/PyEPO](https://khalil-research.github.io/PyEPO).


## Learning Framework

<p align="center"><img width="100%" src="images/learning_framework_e2e.png" /></p>


## Features

- Implement SPO+ [[1]](https://doi.org/10.1287/mnsc.2020.3922) and DBB [[3]](https://arxiv.org/abs/1912.02175)
- Support [Gurobi](https://www.gurobi.com/) and [Pyomo](http://www.pyomo.org/) API
- Support Parallel computing for optimization solver
- Support solution caching [[4]](https://arxiv.org/abs/2011.05354) to speed up training

## Installation

You can download ``PyEPO`` from our GitHub repository.

```bash
git clone https://github.com/khalil-research/PyEPO.git
```

And install it.

```bash
pip install PyEPO/pkg/.
```


## Dependencies

* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)
* [Pathos](https://pathos.readthedocs.io/)
* [tqdm](https://tqdm.github.io/)
* [Pyomo](http://www.pyomo.org/)
* [Gurobi](https://www.gurobi.com/)
* [Scikit Learn](https://scikit-learn.org/)
* [PyTorch](http://pytorch.org/)


## Issue

On Windows system, there is missing ``freeze_support`` to run ``multiprocessing`` directly from ``__main__``. When ``processes`` is not 1, try ``if __name__ == "__main__":`` instead of Jupyter notebook or a PY file.


## Sample Code

```python
#!/usr/bin/env python
# coding: utf-8

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pyepo
from pyepo.model.grb import optGrbModel
import torch
from torch import nn
from torch.utils.data import DataLoader


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


# prediction model
class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_feat, num_item)

    def forward(self, x):
        out = self.linear(x)
        return out


if __name__ == "__main__":

    # generate data
    num_data = 1000 # number of data
    num_feat = 5 # size of feature
    num_item = 10 # number of items
    weights, x, c = pyepo.data.knapsack.genData(num_data, num_feat, num_item,
                                                dim=3, deg=4, noise_width=0.5, seed=135)

    # init optimization model
    optmodel = myModel(weights)

    # init prediction model
    predmodel = LinearRegression()
    # set optimizer
    optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-2)
    # init SPO+ loss
    spop = pyepo.func.SPOPlus(optmodel, processes=1)

    # build dataset
    dataset = pyepo.data.dataset.optDataset(optmodel, x, c)
    # get data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # training
    num_epochs = 10
    for epoch in range(num_epochs):
        for data in dataloader:
            x, c, w, z = data
            # forward pass
            cp = predmodel(x)
            loss = spop(cp, c, w, z).mean()
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # eval
    regret = pyepo.metric.regret(predmodel, optmodel, dataloader)
    print("Regret on Training Set: {:.4f}".format(regret))

```


## Reference
* [1] [Elmachtoub, A. N., & Grigas, P. (2021). Smart “predict, then optimize”. Management Science.](https://doi.org/10.1287/mnsc.2020.3922)
* [2] [Mandi, J., Stuckey, P. J., & Guns, T. (2020). Smart predict-and-optimize for hard combinatorial optimization problems. In Proceedings of the AAAI Conference on Artificial Intelligence.](https://doi.org/10.1609/aaai.v34i02.5521)
* [3] [Vlastelica, M., Paulus, A., Musil, V., Martius, G., & Rolínek, M. (2019). Differentiation of blackbox combinatorial solvers. arXiv preprint arXiv:1912.02175.](https://arxiv.org/abs/1912.02175)
* [4] [Mulamba, Maxime, et al. "Contrastive losses and solution caching for predict-and-optimize." arXiv preprint arXiv:2011.05354 (2020).](https://arxiv.org/abs/2011.05354)
