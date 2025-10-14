from pyepo.predictive.pred import PredictivePrescription
from scipy.spatial import distance
import numpy as np 

class NearestPrediction(PredictivePrescription):

    def __init__(self, feats, costs, k, model):
        self.features = feats
        self.costs = costs
        self.k = k 
        super().__init__(model)

    def _get_weights(self, x):
        dists = distance.cdist([x], self.features, metric="euclidean").flatten()
        idx = np.argpartition(dists, self.k)[:self.k]
        weights = np.zeros(len(self.features))
        weights[idx] = 1.0 / self.k
        return weights