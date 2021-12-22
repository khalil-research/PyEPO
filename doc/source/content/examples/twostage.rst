Two-stage Method
++++++++++++++++

Two-stage approach trains a regression model :math:`\hat{c} = g(x, \theta)` by minimizing a prediction error :math:`l(\hat{c}, c)` such as mean square error :math:`l_{MSE}(\hat{c}, c) = \frac{1}{n} \| \hat{c} - c \| ^ 2`. Then in an inference process, the machine learning model predicts :math:`\hat{c} = g(x, \theta)` first. After that, the predicted value :math:`\hat{c}` is used for solving the optimization problem.

.. autofunction:: spo.twostage.sklearnPred

``spo.twostage.sklearnPred`` is a helper function to build multi-output regressor with scikit-learn.

.. code-block:: python

   import spo

   # model for shortest path
   grid = (5,5) # grid size
   model = spo.model.grb.shortestPathModel(grid)

   # generate data
   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   deg = 4 # polynomial degree
   noise_width = 0 # noise width
   x, c = spo.data.shortestpath.genData(num_data, num_feat, grid, deg, noise_width, seed=135)

   # sklearn regressor
   from sklearn.linear_model import LinearRegression
   reg = LinearRegression() # linear regression

   # build model
   twostage_model = spo.twostage.sklearnPred(reg)

   # training
   twostage_model.fit(x, c)

   # prediction
   c_pred = twostage_model.predict(x)
