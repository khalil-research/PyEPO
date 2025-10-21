from pyepo.predictive.pred import PredictivePrescription
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pyepo import EPO

class RandomForestPrescription(PredictivePrescription):
    def __init__(self, feats, costs, model, random_state=None):
        super().__init__(model)
        self.features = feats
        self.costs = costs
        self.random_state = random_state

        # optimize model parameters automatically
        self.weigth_model = self._optimize_model()

    def _get_weights(self, x):
        T = len(self.weigth_model.estimators_)
        N = len(self.features)
        weights = np.zeros(N)

        for tree in self.weigth_model.estimators_:
            leaf_x = tree.apply([x])[0]
            leaf_train = tree.apply(self.features)
            same_leaf = (leaf_train == leaf_x)
            idx = np.where(same_leaf)[0]
            if len(idx) > 0:
                weights[idx] += 1.0 / (T * len(idx))
        return weights / np.sum(weights)

    def _optimize_model(self):
        # split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            self.features, self.costs, test_size=0.2, random_state=self.random_state
        )

        # search grid (reasonable defaults)
        n_estimators_list = [50, 100, 200]
        max_depth_list = [5, 10, 20, None]
        # min_samples_leaf_list = [1, 2, 5]

        best_score = np.inf
        best_params = None

        for n_est in n_estimators_list:
            for depth in max_depth_list:
            # for min_leaf in min_samples_leaf_list:
                model = RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth=depth,
                    # min_samples_leaf=min_leaf,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_val)

                loss = 0
                optsum = 0
                for pred, true_cost in zip(preds, y_val):
                    self.model.setObj(true_cost)
                    _, true_obj = self.model.solve()

                    pred_obj = self.model.cal_obj(true_cost, pred)

                    if self.model.modelSense == EPO.MINIMIZE:
                        loss += pred_obj - true_obj
                    if self.model.modelSense == EPO.MAXIMIZE:
                        loss += true_obj - pred_obj

                    optsum += abs(true_obj)

                score = loss/(optsum + 1e-7)

                if score < best_score:
                    best_score = score
                    best_params = {
                        "n_estimators": n_est,
                        "max_depth": depth,
                        # "min_samples_leaf": min_leaf,
                    }

        # retrain on full data with best params
        best_model = RandomForestRegressor(**best_params, random_state=self.random_state, n_jobs=-1)
        best_model.fit(self.features, self.costs)

        return best_model