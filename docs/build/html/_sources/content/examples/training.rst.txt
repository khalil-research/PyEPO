Training
++++++++

``pyepo.func`` is extended from PyTorch modules to support automatic differentiation. Users can train neural network for end-to-end predict-then-optimize problem.

Training with SPO+
==================

The example to learn shortest path with linear model is as follows:

.. code-block:: python

   import pyepo
   import torch
   from torch import nn
   from torch.utils.data import DataLoader

   # model for shortest path
   grid = (5,5) # grid size
   optmodel = pyepo.model.grb.shortestPathModel(grid)

   # generate data
   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   deg = 4 # polynomial degree
   noise_width = 0.5 # noise width
   x, c = pyepo.data.shortestpath.genData(num_data, num_feat, grid, deg, noise_width, seed=135)

   # build dataset
   dataset = pyepo.data.dataset.optDataset(optmodel, x, c)

   # get data loader
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

   # build linear model
   class LinearRegression(nn.Module):

       def __init__(self):
           super(LinearRegression, self).__init__()
           self.linear = nn.Linear(5, 40)

       def forward(self, x):
           out = self.linear(x)
           return out
   # init
   predmodel = LinearRegression()
   # set optimizer
   optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-3)
   # init SPO+ loss
   spo = pyepo.func.SPOPlus(optmodel, processes=8)

   # training
   num_epochs = 100
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           # forward pass
           cp = predmodel(x)
           loss = spo(cp, c, w, z).mean()
           # backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Training with DBB
=================

The example to learn shortest path with linear model is as follows:

.. code-block:: python

   import pyepo
   import torch
   from torch import nn
   from torch.utils.data import DataLoader

   # model for shortest path
   grid = (5,5) # grid size
   optmodel = pyepo.model.grb.shortestPathModel(grid)

   # generate data
   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   deg = 4 # polynomial degree
   noise_width = 0.5 # noise width
   x, c = pyepo.data.shortestpath.genData(num_data, num_feat, grid, deg, noise_width, seed=135)

   # build dataset
   dataset = pyepo.data.dataset.optDataset(optmodel, x, c)

   # get data loader
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

   # build linear model
   class LinearRegression(nn.Module):

       def __init__(self):
           super(LinearRegression, self).__init__()
           self.linear = nn.Linear(5, 40)

       def forward(self, x):
           out = self.linear(x)
           return out
   # init
   predmodel = LinearRegression()
   # set optimizer
   optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-3)
   # init black-box
   dbb = pyepo.func.blackboxOpt(optmodel, lambd=10, processes=8)
   # init loss
   criterion = nn.L1Loss()

   # training
   num_epochs = 100
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           # forward pass
           cp = predmodel(x)
           # black-box optimizer
           wp = dbb(cp)
           # objective value
           zp = (wp * c).sum(1).view(-1, 1)
           # regret loss
           loss = criterion(zp, z)
           # backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Training with DPO
=================

The example to learn shortest path with linear model is as follows:

.. code-block:: python

   import pyepo
   import torch
   from torch import nn
   from torch.utils.data import DataLoader

   # model for shortest path
   grid = (5,5) # grid size
   optmodel = pyepo.model.grb.shortestPathModel(grid)

   # generate data
   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   deg = 4 # polynomial degree
   noise_width = 0.5 # noise width
   x, c = pyepo.data.shortestpath.genData(num_data, num_feat, grid, deg, noise_width, seed=135)

   # build dataset
   dataset = pyepo.data.dataset.optDataset(optmodel, x, c)

   # get data loader
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

   # build linear model
   class LinearRegression(nn.Module):

       def __init__(self):
           super(LinearRegression, self).__init__()
           self.linear = nn.Linear(5, 40)

       def forward(self, x):
           out = self.linear(x)
           return out
   # init
   predmodel = LinearRegression()
   # set optimizer
   optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-3)
   # init perturbed optimizer
   ptb = pyepo.func.perturbedOpt(optmodel, n_samples=10, epsilon=0.5, processes=8)
   # init loss
   criterion = nn.MSELoss()

   # training
   num_epochs = 100
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           # forward pass
           cp = predmodel(x)
           # perturbed optimizer
           we = ptb(cp)
           # loss
           loss = criterion(we, w)
           # backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Training with PFYL
==================

The example to learn shortest path with linear model is as follows:

.. code-block:: python

   import pyepo
   import torch
   from torch import nn
   from torch.utils.data import DataLoader

   # model for shortest path
   grid = (5,5) # grid size
   optmodel = pyepo.model.grb.shortestPathModel(grid)

   # generate data
   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   deg = 4 # polynomial degree
   noise_width = 0.5 # noise width
   x, c = pyepo.data.shortestpath.genData(num_data, num_feat, grid, deg, noise_width, seed=135)

   # build dataset
   dataset = pyepo.data.dataset.optDataset(optmodel, x, c)

   # get data loader
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

   # build linear model
   class LinearRegression(nn.Module):

       def __init__(self):
           super(LinearRegression, self).__init__()
           self.linear = nn.Linear(5, 40)

       def forward(self, x):
           out = self.linear(x)
           return out
   # init
   predmodel = LinearRegression()
   # set optimizer
   optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-3)
   # init PFY loss
   pfy = pyepo.func.perturbedFenchelYoung(optmodel, n_samples=10, epsilon=0.5, processes=8)

   # training
   num_epochs = 100
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           # forward pass
           cp = predmodel(x)
           # Fenchel-Young loss
           loss = pfy(cp, w).mean()
           # backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
