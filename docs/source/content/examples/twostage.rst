Two-stage Method
++++++++++++++++

Two-stage approach trains a regression model :math:`\hat{c} = g(\mathbf{x}; \mathbf{\theta})` by minimizing a prediction error :math:`l(\hat{\mathbf{c}}, \mathbf{c})` such as mean square error :math:`l_{MSE}(\hat{\mathbf{c}}, \mathbf{c}) = \frac{1}{n} \sum_i^n \| \hat{\mathbf{c}}_i - \mathbf{c}_i \| ^ 2`. Then in an inference process, the machine learning model predicts :math:`\hat{c} = g(\mathbf{x}; \mathbf{\theta})` first. After that, the predicted value :math:`\hat{c}` is used for solving the optimization problem.

.. autofunction:: pyepo.twostage.sklearnPred
    :noindex:

``pyepo.twostage.sklearnPred`` is a helper function to build multi-output regressor with scikit-learn.

.. code-block:: python

   import pyepo

   # model for shortest path
   grid = (5,5) # grid size
   model = pyepo.model.grb.shortestPathModel(grid)

   # generate data
   num_data = 1000 # number of data
   num_feat = 5 # size of feature
   deg = 4 # polynomial degree
   noise_width = 0 # noise width
   x, c = pyepo.data.shortestpath.genData(num_data, num_feat, grid, deg, noise_width, seed=135)

   # sklearn regressor
   from sklearn.linear_model import LinearRegression
   reg = LinearRegression() # linear regression

   # build model
   twostage_model = pyepo.twostage.sklearnPred(reg)

   # training
   twostage_model.fit(x, c)

   # prediction
   c_pred = twostage_model.predict(x)
