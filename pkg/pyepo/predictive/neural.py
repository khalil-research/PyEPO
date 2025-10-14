from pyepo.predictive.pred import PredictivePrescription
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from pyepo.func.surrogate import SPOPlus
from pyepo.model.opt import optModel
from tqdm import tqdm
import numpy as np
import copy

class NeuralPrediction(PredictivePrescription):

    def __init__(self, feats, costs, weight_model, model):
        self.features = feats
        self.costs = costs
        self.weight_model: nn.Module = weight_model
        super().__init__(model)

    def _get_weights(self, x, features=None):
        if features is None:
            features = self.features

        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if not torch.is_tensor(features):
            features = torch.tensor(features, dtype=torch.float32)

        if x.dim() == 1:
            x = x.unsqueeze(0)           # [1, D]
        if features.dim() == 2:
            features = features.unsqueeze(0)  # [1, N, D]

        x = x.to(self.weight_model.net[0].weight.device)
        features = features.to(self.weight_model.net[0].weight.device)

        weights = self.weight_model(x, features)
        return weights
    

    def train_model(self, epochs=100, batch_size=32, lr=1e-3, val_split=0.2):
        X_train, X_val, y_train, y_val = train_test_split(
            self.features, self.costs, test_size=val_split, random_state=0
        )

        optimizer = optim.Adam(self.weight_model.parameters(), lr=lr)
        loss_fn = SPOPlus(self.model, processes=1)

        train_loader = torch.utils.data.DataLoader(
            LeaveOneOutDataset(self.model, X_train, y_train),
            batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            LeaveOneOutDataset(self.model, X_val, y_val),
            batch_size=batch_size, shuffle=False
        )

        early_stopper = EarlyStopper(5, 0.05)

        for epoch in range(epochs):
            self.weight_model.train()
            train_loss = 0.0
            for i, data in enumerate(train_loader):
                x, c, feats_batch, costs_batch, w, z = data  

                if torch.cuda.is_available():
                    x, c, feats_batch, costs_batch, w, z = x.cuda(), c.cuda(), feats_batch.cuda(), costs_batch.cuda(), w.cuda(), z.cuda()

                weights = self._get_weights(x, feats_batch)             # [B, N]
                preds = (weights.unsqueeze(-1) * costs_batch).sum(dim=1)  # [B, C] # TODO: make this more abstract and let it work with multiple obj function
                loss = loss_fn(preds, c, w, z)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(x)

            train_loss /= len(train_loader.dataset)

            # validation
            self.weight_model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for i, data in enumerate(val_loader):
                    x, c, feats_batch, costs_batch, w, z = data  

                    if torch.cuda.is_available():
                        x, c, feats_batch, costs_batch, w, z = x.cuda(), c.cuda(), feats_batch.cuda(), costs_batch.cuda(), w.cuda(), z.cuda()

                    feats_batch = torch.FloatTensor(X_train).cuda()
                    y_batch = torch.FloatTensor(y_train).cuda()
                    feats_batch = feats_batch.unsqueeze(0).expand(len(x), -1, -1).contiguous()  # [B, N, D]
                    y_batch = y_batch.unsqueeze(0).expand(len(costs_batch), -1, -1).contiguous()
                    weights = self._get_weights(x, feats_batch)

                    preds = (weights.unsqueeze(-1) * y_batch).sum(dim=1) 
                    val_loss += loss_fn(preds, c, w, z).item() * len(x)

                val_loss /= len(val_loader.dataset)
            print(f"Epoch {epoch+1:03d}: train={train_loss:.4f}, val={val_loss:.4f}")
            
            if early_stopper.step(val_loss, self.weight_model):
                print(f"Early stopping at epoch {epoch+1}. Restored best weights.")
                break

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_state_dict = None

    def step(self, validation_loss, model):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
            return False 
        else:
            self.counter += 1
            if self.counter >= self.patience:
                # restore best weights and stop
                if self.best_state_dict is not None:
                    model.load_state_dict(self.best_state_dict)
                return True  # stop training
            return False    


class LeaveOneOutDataset(torch.utils.data.Dataset):
    """
    This class is Torch Dataset for optimization problems.

    Attributes:
        model (optModel): Optimization models
        feats (np.ndarray): Data features
        costs (np.ndarray): Cost vectors
        sols (np.ndarray): Optimal solutions
        objs (np.ndarray): Optimal objective values
    """

    def __init__(self, model, feats, costs):
        """
        A method to create a optDataset from optModel

        Args:
            model (optModel): an instance of optModel
            feats (np.ndarray): data features
            costs (np.ndarray): costs of objective function
        """
        if not isinstance(model, optModel):
            raise TypeError("arg model is not an optModel")
        self.model = model
        # data
        self.feats = feats
        self.costs = costs
        # find optimal solutions
        self.sols, self.objs = self._getSols()

    def _getSols(self):
        """
        A method to get optimal solutions for all cost vectors
        """
        sols = []
        objs = []
        print("\nOptimizing for optDataset...", flush=True)
        for c in tqdm(self.costs):
            try:
                sol, obj = self._solve(c)
                # to numpy
                if isinstance(sol, torch.Tensor):
                    sol = sol.detach().cpu().numpy()
            except:
                raise ValueError(
                    "For optModel, the method 'solve' should return solution vector and objective value."
                )
            sols.append(sol)
            objs.append([obj])
        return np.array(sols), np.array(objs)

    def _solve(self, cost):
        """
        A method to solve optimization problem to get an optimal solution with given cost

        Args:
            cost (np.ndarray): cost of objective function

        Returns:
            tuple: optimal solution (np.ndarray) and objective value (float)
        """
        self.model.setObj(cost)
        sol, obj = self.model.solve()
        return sol, obj

    def __len__(self):
        """
        A method to get data size

        Returns:
            int: the number of optimization problems
        """
        return len(self.costs)

    def __getitem__(self, index):
        """
        A method to retrieve data

        Args:
            index (int): data index

        Returns:
            tuple: data features (torch.tensor), costs (torch.tensor), optimal solutions (torch.tensor) and objective values (torch.tensor)
        """

        x_i = self.feats[index]
        c_i = self.costs[index]

        mask = torch.ones(len(self.feats), dtype=torch.bool)
        mask[index] = False
        X_rest = self.feats[mask]
        C_rest = self.costs[mask]

        return (
            torch.FloatTensor(x_i), 
            torch.FloatTensor(c_i), 
            torch.FloatTensor(X_rest), 
            torch.FloatTensor(C_rest),
            torch.FloatTensor(self.sols[index]),
            torch.FloatTensor(self.objs[index]),
        )