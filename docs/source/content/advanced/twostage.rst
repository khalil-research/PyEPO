Two-stage Baseline
++++++++++++++++++

The two-stage approach trains a regression model
:math:`\hat{\mathbf{c}} = g(\mathbf{x}; \boldsymbol{\theta})` with a
prediction loss such as mean squared error. At inference time, the model first
predicts :math:`\hat{\mathbf{c}}`, then the predicted costs are used to solve
the optimization problem.

The two-stage model does not train through the optimization model, but it is
evaluated with the same decision-quality metrics as end-to-end methods.


When to Use It
==============

Use a two-stage baseline when:

* true cost labels are available;
* the prediction model can be trained with a standard regression loss;
* the experiment needs a non-end-to-end baseline.

After training, evaluate the predictor with ``pyepo.metric.regret`` or
``pyepo.metric.unambRegret``.

.. autofunction:: pyepo.twostage.sklearnPred
    :noindex:

``pyepo.twostage.sklearnPred`` is a helper function that wraps a scikit-learn estimator into a multi-output regressor.


Minimal Example
===============

.. code-block:: python

   import pyepo
   from sklearn.linear_model import LinearRegression
   from torch.utils.data import DataLoader

   grid = (5, 5)
   model = pyepo.model.shortestPathModel(grid)

   feat, costs = pyepo.data.shortestpath.genData(
       num_data=1000,
       num_features=5,
       grid=grid,
       deg=4,
       noise_width=0.5,
       seed=135,
   )

   reg = LinearRegression()
   twostage_model = pyepo.twostage.sklearnPred(reg)
   twostage_model.fit(feat, costs)

   dataset = pyepo.data.dataset.optDataset(model, feat, costs)
   dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
   # regret takes a callable mapping features to costs; pass the predictor's .predict
   regret = pyepo.metric.regret(twostage_model.predict, model, dataloader)
