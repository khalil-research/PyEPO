Training
++++++++

``spo.func.SPOPlus`` and ``spo.func.blackboxOpt`` are PyTorch modules to support automatic differentiation. Users can train neural network for end-to-end predict-then-optimize problem.

Training with SPO+
==================

The example to learn shortest path with linear model is as follows:

.. code-block:: python

   import spo
   import torch
   from torch import nn
   from torch.utils.data import DataLoader

   # model for shortest path
   grid = (5,5) # grid size
   model = spo.model.grb.shortestPathModel(grid)

   # generate data
   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   deg = 4 # polynomial degree
   noise_width = 0.5 # noise width
   x, c = spo.data.shortestpath.genData(num_data, num_feat, grid, deg, noise_width, seed=135)

   # build dataset
   dataset = spo.data.dataset.optDataset(model, x, c)

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
   reg = LinearRegression()
   # set optimizer
   optimizer = torch.optim.Adam(reg.parameters(), lr=1e-3)
   # init SPO+ loss
   spo_func = spo.func.SPOPlus(model, processes=8)

   # training
   num_epochs = 100
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           # forward pass
           cp = reg(x)
           loss = spo_func.apply(cp, c, w, z).mean()
           # backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()


Training with DBB
=================

The example to learn shortest path with linear model is as follows:

.. code-block:: python

   import spo
   import torch
   from torch import nn
   from torch.utils.data import DataLoader

   # model for shortest path
   grid = (5,5) # grid size
   model = spo.model.grb.shortestPathModel(grid)

   # generate data
   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   deg = 4 # polynomial degree
   noise_width = 0.5 # noise width
   x, c = spo.data.shortestpath.genData(num_data, num_feat, grid, deg, noise_width, seed=135)

   # build dataset
   dataset = spo.data.dataset.optDataset(model, x, c)

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
   reg = LinearRegression()
   # set optimizer
   optimizer = torch.optim.Adam(reg.parameters(), lr=1e-3)
   # init SPO+ loss
   dbb_optm = spo.func.blackboxOpt(model, lambd=10, processes=8)
   criterion = nn.L1Loss()

   # training
   num_epochs = 100
   for epoch in range(num_epochs):
       for data in dataloader:
           x, c, w, z = data
           # forward pass
           cp = reg(x)
           # black-box optimizer
           wp = dbb_optm.apply(cp)
           # objective value
           zp = (wp * c).sum(1).view(-1, 1)
           # loss
           loss = criterion(zp, z)
           # backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
