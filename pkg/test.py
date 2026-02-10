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
        # create a model
        m = gp.Model()
        # variables
        x = m.addVars(self.num_item, name="x", vtype=GRB.BINARY)
        # sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(gp.quicksum([self.weights[0,i] * x[i] for i in range(self.num_item)]) <= 7)
        m.addConstr(gp.quicksum([self.weights[1,i] * x[i] for i in range(self.num_item)]) <= 8)
        m.addConstr(gp.quicksum([self.weights[2,i] * x[i] for i in range(self.num_item)]) <= 9)
        return m, x


# prediction model
class LinearRegression(nn.Module):

    def __init__(self, num_feat, num_item):
        super().__init__()
        self.linear = nn.Linear(num_feat, num_item)

    def forward(self, x):
        out = self.linear(x)
        return out


def test_model_ops(name, optmodel, cost):
    """Test copy, solve, and addConstr for a given model."""
    print("\n=== Model Ops: {} ===".format(name))

    # solve
    optmodel.setObj(cost)
    sol, obj = optmodel.solve()
    print("  Obj: {:.4f}  Sol sum: {:.4f}".format(obj, np.sum(sol)))

    # copy
    copied = optmodel.copy()
    copied.setObj(cost)
    sol2, obj2 = copied.solve()
    assert np.isclose(obj, obj2, atol=1e-3), "copy() changed objective: {} vs {}".format(obj, obj2)
    print("  copy() OK: obj matches")

    # addConstr
    coefs = [1] * optmodel.num_cost
    rhs = optmodel.num_cost * 0.5
    constrained = optmodel.addConstr(coefs, rhs)
    constrained.setObj(cost)
    sol3, obj3 = constrained.solve()
    print("  addConstr() OK: new obj = {:.4f}".format(obj3))

    return True


def test_shortestpath_backend(name, sp_model, num_arcs):
    """Test shortestpath model for a given backend."""
    print("\n=== ShortestPath: {} ===".format(name))
    cost = np.random.RandomState(42).rand(num_arcs)

    sp_model.setObj(cost)
    sol, obj = sp_model.solve()
    print("  Obj: {:.4f}".format(obj))

    # copy
    copied = sp_model.copy()
    copied.setObj(cost)
    sol2, obj2 = copied.solve()
    assert np.isclose(obj, obj2, atol=1e-3), "copy() mismatch: {} vs {}".format(obj, obj2)
    print("  copy() OK")

    # addConstr
    constrained = sp_model.addConstr([1] * num_arcs, num_arcs * 0.5)
    constrained.setObj(cost)
    sol3, obj3 = constrained.solve()
    print("  addConstr() OK: new obj = {:.4f}".format(obj3))


def test_portfolio_backend(name, portfolio_model, num_assets):
    """Test portfolio model for a given backend."""
    print("\n=== Portfolio: {} ===".format(name))
    cost = np.random.RandomState(42).rand(num_assets)

    portfolio_model.setObj(cost)
    sol, obj = portfolio_model.solve()
    print("  Obj: {:.4f}".format(obj))

    # copy
    copied = portfolio_model.copy()
    copied.setObj(cost)
    sol2, obj2 = copied.solve()
    assert np.isclose(obj, obj2, atol=1e-3), "copy() mismatch: {} vs {}".format(obj, obj2)
    print("  copy() OK")

    # addConstr (restrict first 3 assets only to avoid infeasibility with budget constraint)
    coefs = np.zeros(portfolio_model.num_cost)
    coefs[:3] = 1.0
    constrained = portfolio_model.addConstr(coefs, 0.5)
    constrained.setObj(cost)
    sol3, obj3 = constrained.solve()
    print("  addConstr() OK: new obj = {:.4f}".format(obj3))


def test_tsp_backend(name, tsp_model, num_edges):
    """Test TSP model for a given backend."""
    print("\n=== TSP: {} ===".format(name))
    cost = np.random.RandomState(42).rand(num_edges)

    tsp_model.setObj(cost)
    sol, obj = tsp_model.solve()
    print("  Obj: {:.4f}  Sol sum: {:.4f}".format(obj, np.sum(sol)))

    # getTour
    tour = tsp_model.getTour(sol)
    print("  Tour: {}".format(tour))

    # copy
    copied = tsp_model.copy()
    copied.setObj(cost)
    sol2, obj2 = copied.solve()
    assert np.isclose(obj, obj2, atol=1e-3), "copy() mismatch: {} vs {}".format(obj, obj2)
    print("  copy() OK")

    # addConstr
    constrained = tsp_model.addConstr([1] * num_edges, num_edges - 1)
    constrained.setObj(cost)
    sol3, obj3 = constrained.solve()
    print("  addConstr() OK: new obj = {:.4f}".format(obj3))

    return obj


if __name__ == "__main__":

    # generate data
    num_data = 1000 # number of data
    num_feat = 5 # size of feature
    num_item = 10 # number of items
    weights, x, c = pyepo.data.knapsack.genData(num_data, num_feat, num_item, dim=3, deg=4, noise_width=0.1, seed=135)

    # init optimization model
    optmodel = myModel(weights)

    # build dataset
    dataset = pyepo.data.dataset.optDataset(optmodel, x, c)
    # get data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # training config
    num_epochs = 10

    def train_and_eval(name, loss_fn, call, lr=1e-2):
        """Train with a given loss function and print results."""
        pred = LinearRegression(num_feat, num_item)
        opt = torch.optim.Adam(pred.parameters(), lr=lr)
        print("\n--- {} ---".format(name))
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            for data in dataloader:
                x, c, w, z = data
                cp = pred(x)
                loss = call(loss_fn, cp, c, w, z)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                num_batches += 1
            avg_loss = epoch_loss / num_batches
            print("  Epoch {:3d}/{}: avg loss = {:.4f}".format(
                epoch + 1, num_epochs, avg_loss))
        reg = pyepo.metric.regret(pred, optmodel, dataloader)
        ms = pyepo.metric.MSE(pred, dataloader)
        print("  Regret: {:.4f}  MSE: {:.4f}".format(reg, ms))

    # ============================================================
    # Part A: Model operations (copy, addConstr) across backends
    # ============================================================
    print("\n" + "="*60)
    print("Part A: Model Operations")
    print("="*60)

    cost_knapsack = np.random.RandomState(42).rand(num_item)
    capacity = np.array([7, 8, 9])  # matches myModel constraints
    grid = (5, 5)
    num_arcs = 40  # (5-1)*5 + (5-1)*5

    # A1. Gurobi knapsack (custom model)
    test_model_ops("Gurobi custom knapsack", optmodel, cost_knapsack)

    # A2. Gurobi built-in knapsack
    from pyepo.model.grb.knapsack import knapsackModel as grbKnapsack
    grb_ks = grbKnapsack(weights, capacity)
    test_model_ops("Gurobi knapsack", grb_ks, cost_knapsack)

    # A3. Gurobi shortestpath
    from pyepo.model.grb.shortestpath import shortestPathModel as grbSP
    test_shortestpath_backend("Gurobi", grbSP(grid), num_arcs)

    # A4. Pyomo backend
    try:
        from pyepo.model.omo.knapsack import knapsackModel as omoKnapsack
        from pyepo.model.omo.shortestpath import shortestPathModel as omoSP

        omo_ks = omoKnapsack(weights, capacity, solver="gurobi")
        test_model_ops("Pyomo knapsack", omo_ks, cost_knapsack)
        test_shortestpath_backend("Pyomo", omoSP(grid, solver="gurobi"), num_arcs)
    except ImportError as e:
        print("\n  [SKIP] Pyomo not available: {}".format(e))

    # A5. COPT backend
    try:
        from pyepo.model.copt.knapsack import knapsackModel as coptKnapsack
        from pyepo.model.copt.shortestpath import shortestPathModel as coptSP

        copt_ks = coptKnapsack(weights, capacity)
        test_model_ops("COPT knapsack", copt_ks, cost_knapsack)
        test_shortestpath_backend("COPT", coptSP(grid), num_arcs)
    except ImportError as e:
        print("\n  [SKIP] COPT not available: {}".format(e))

    # A6. MPAX backend
    try:
        from pyepo.model.mpax.shortestpath import shortestPathModel as mpaxSP

        mpax_sp = mpaxSP(grid)
        test_shortestpath_backend("MPAX", mpax_sp, num_arcs)
    except (ImportError, NameError) as e:
        print("\n  [SKIP] MPAX not available: {}".format(e))

    # A7. Cross-backend consistency (Gurobi vs Pyomo vs COPT shortestpath)
    print("\n=== Cross-backend ShortestPath Consistency ===")
    sp_cost = np.random.RandomState(99).rand(num_arcs)
    grb_sp = grbSP(grid)
    grb_sp.setObj(sp_cost)
    _, grb_obj = grb_sp.solve()
    print("  Gurobi obj: {:.6f}".format(grb_obj))
    try:
        omo_sp = omoSP(grid, solver="gurobi")
        omo_sp.setObj(sp_cost)
        _, omo_obj = omo_sp.solve()
        assert np.isclose(grb_obj, omo_obj, atol=1e-3), "Pyomo mismatch"
        print("  Pyomo  obj: {:.6f} (matches)".format(omo_obj))
    except Exception as e:
        print("  Pyomo: {}".format(e))
    try:
        copt_sp = coptSP(grid)
        copt_sp.setObj(sp_cost)
        _, copt_obj = copt_sp.solve()
        assert np.isclose(grb_obj, copt_obj, atol=1e-3), "COPT mismatch"
        print("  COPT   obj: {:.6f} (matches)".format(copt_obj))
    except Exception as e:
        print("  COPT: {}".format(e))
    try:
        mpax_sp2 = mpaxSP(grid)
        mpax_sp2.setObj(sp_cost)
        _, mpax_obj = mpax_sp2.solve()
        assert np.isclose(grb_obj, mpax_obj, atol=1e-2), "MPAX mismatch"
        print("  MPAX   obj: {:.6f} (matches)".format(mpax_obj))
    except Exception as e:
        print("  MPAX: {}".format(e))

    # A8. Gurobi portfolio
    num_assets = 50
    covariance = np.random.RandomState(42).rand(num_assets, num_assets)
    covariance = covariance @ covariance.T / num_assets  # make PSD
    from pyepo.model.grb.portfolio import portfolioModel as grbPortfolio
    grb_pf = grbPortfolio(num_assets, covariance)
    test_portfolio_backend("Gurobi", grb_pf, num_assets)

    # A9. Gurobi TSP
    num_nodes = 5
    num_edges = num_nodes * (num_nodes - 1) // 2
    tsp_cost = np.random.RandomState(42).rand(num_edges)
    from pyepo.model.grb.tsp import tspGGModel as grbTspGG, tspMTZModel as grbTspMTZ, tspDFJModel as grbTspDFJ
    test_tsp_backend("Gurobi TSP-GG", grbTspGG(num_nodes), num_edges)
    test_tsp_backend("Gurobi TSP-MTZ", grbTspMTZ(num_nodes), num_edges)
    test_tsp_backend("Gurobi TSP-DFJ", grbTspDFJ(num_nodes), num_edges)

    # A10. Pyomo portfolio + TSP
    try:
        from pyepo.model.omo.portfolio import portfolioModel as omoPortfolio
        from pyepo.model.omo.tsp import tspGGModel as omoTspGG, tspMTZModel as omoTspMTZ

        omo_pf = omoPortfolio(num_assets, covariance, solver="gurobi")
        test_portfolio_backend("Pyomo", omo_pf, num_assets)
        test_tsp_backend("Pyomo TSP-GG", omoTspGG(num_nodes, solver="gurobi"), num_edges)
        test_tsp_backend("Pyomo TSP-MTZ", omoTspMTZ(num_nodes, solver="gurobi"), num_edges)
    except ImportError as e:
        print("\n  [SKIP] Pyomo portfolio/TSP not available: {}".format(e))

    # A11. COPT portfolio + TSP
    try:
        from pyepo.model.copt.portfolio import portfolioModel as coptPortfolio
        from pyepo.model.copt.tsp import tspGGModel as coptTspGG, tspMTZModel as coptTspMTZ, tspDFJModel as coptTspDFJ

        copt_pf = coptPortfolio(num_assets, covariance)
        test_portfolio_backend("COPT", copt_pf, num_assets)
        test_tsp_backend("COPT TSP-GG", coptTspGG(num_nodes), num_edges)
        test_tsp_backend("COPT TSP-MTZ", coptTspMTZ(num_nodes), num_edges)
        test_tsp_backend("COPT TSP-DFJ", coptTspDFJ(num_nodes), num_edges)
    except ImportError as e:
        print("\n  [SKIP] COPT portfolio/TSP not available: {}".format(e))

    # A12. Cross-backend portfolio consistency
    print("\n=== Cross-backend Portfolio Consistency ===")
    pf_cost = np.random.RandomState(99).rand(num_assets)
    grb_pf2 = grbPortfolio(num_assets, covariance)
    grb_pf2.setObj(pf_cost)
    _, grb_pf_obj = grb_pf2.solve()
    print("  Gurobi obj: {:.6f}".format(grb_pf_obj))
    try:
        omo_pf2 = omoPortfolio(num_assets, covariance, solver="gurobi")
        omo_pf2.setObj(pf_cost)
        _, omo_pf_obj = omo_pf2.solve()
        assert np.isclose(grb_pf_obj, omo_pf_obj, atol=1e-3), "Pyomo portfolio mismatch"
        print("  Pyomo  obj: {:.6f} (matches)".format(omo_pf_obj))
    except Exception as e:
        print("  Pyomo: {}".format(e))
    try:
        copt_pf2 = coptPortfolio(num_assets, covariance)
        copt_pf2.setObj(pf_cost)
        _, copt_pf_obj = copt_pf2.solve()
        assert np.isclose(grb_pf_obj, copt_pf_obj, atol=1e-3), "COPT portfolio mismatch"
        print("  COPT   obj: {:.6f} (matches)".format(copt_pf_obj))
    except Exception as e:
        print("  COPT: {}".format(e))

    # A13. Cross-backend TSP consistency
    print("\n=== Cross-backend TSP Consistency ===")
    grb_gg = grbTspGG(num_nodes)
    grb_gg.setObj(tsp_cost)
    _, grb_tsp_obj = grb_gg.solve()
    print("  Gurobi GG obj: {:.6f}".format(grb_tsp_obj))
    try:
        omo_gg = omoTspGG(num_nodes, solver="gurobi")
        omo_gg.setObj(tsp_cost)
        _, omo_tsp_obj = omo_gg.solve()
        assert np.isclose(grb_tsp_obj, omo_tsp_obj, atol=1e-3), "Pyomo TSP mismatch"
        print("  Pyomo  GG obj: {:.6f} (matches)".format(omo_tsp_obj))
    except Exception as e:
        print("  Pyomo: {}".format(e))
    try:
        copt_gg = coptTspGG(num_nodes)
        copt_gg.setObj(tsp_cost)
        _, copt_tsp_obj = copt_gg.solve()
        assert np.isclose(grb_tsp_obj, copt_tsp_obj, atol=1e-3), "COPT TSP mismatch"
        print("  COPT   GG obj: {:.6f} (matches)".format(copt_tsp_obj))
    except Exception as e:
        print("  COPT: {}".format(e))

    # ============================================================
    # Part B: Training with different loss functions (Gurobi)
    # ============================================================
    print("\n" + "="*60)
    print("Part B: Training with Loss Functions")
    print("="*60)

    # 1. SPO+ (surrogate)
    spo = pyepo.func.SPOPlus(optmodel, processes=4)
    train_and_eval("SPOPlus", spo,
                   lambda fn, cp, c, w, z: fn(cp, c, w, z))

    # 2. Perturbation Gradient (surrogate)
    pg = pyepo.func.perturbationGradient(optmodel, processes=1, sigma=1.0)
    train_and_eval("perturbationGradient", pg,
                   lambda fn, cp, c, w, z: fn(cp, c))

    # task loss for methods that return solutions (MAXIMIZE: minimize -c^T w_hat)
    def task_loss(fn, cp, c, w, z):
        w_hat = fn(cp)
        return -(c * w_hat).sum(dim=1).mean()

    # 3. Blackbox Differentiable Optimizer
    bb = pyepo.func.blackboxOpt(optmodel, processes=4, lambd=10)
    train_and_eval("blackboxOpt", bb, task_loss)

    # 4. Negative Identity
    nid = pyepo.func.negativeIdentity(optmodel, processes=1)
    train_and_eval("negativeIdentity", nid, task_loss)

    # 5. Perturbed Optimizer
    ptb = pyepo.func.perturbedOpt(optmodel, processes=4, n_samples=5, sigma=1.0)
    train_and_eval("perturbedOpt", ptb, task_loss)

    # 6. Perturbed Fenchel-Young
    pfy = pyepo.func.perturbedFenchelYoung(optmodel, processes=1, n_samples=5, sigma=1.0)
    train_and_eval("perturbedFenchelYoung", pfy,
                   lambda fn, cp, c, w, z: fn(cp, w))

    # 7. Implicit MLE
    imle = pyepo.func.implicitMLE(optmodel, processes=4, n_samples=5, sigma=1.0)
    train_and_eval("implicitMLE", imle, task_loss)

    # 8. Adaptive Implicit MLE
    aimle = pyepo.func.adaptiveImplicitMLE(optmodel, processes=1, n_samples=5, sigma=1.0)
    train_and_eval("adaptiveImplicitMLE", aimle, task_loss)

    # 9. NCE (contrastive)
    nce = pyepo.func.NCE(optmodel, processes=4, solve_ratio=1, dataset=dataset)
    train_and_eval("NCE", nce,
                   lambda fn, cp, c, w, z: fn(cp, w))

    # 10. Contrastive MAP
    cmap = pyepo.func.contrastiveMAP(optmodel, processes=1, solve_ratio=1, dataset=dataset)
    train_and_eval("contrastiveMAP", cmap,
                   lambda fn, cp, c, w, z: fn(cp, w))

    # 11. Listwise Learning-to-Rank
    ltr = pyepo.func.listwiseLTR(optmodel, processes=4, solve_ratio=1, dataset=dataset)
    train_and_eval("listwiseLTR", ltr,
                   lambda fn, cp, c, w, z: fn(cp, c))

    # 12. Pairwise Learning-to-Rank
    pw = pyepo.func.pairwiseLTR(optmodel, processes=1, solve_ratio=1, dataset=dataset)
    train_and_eval("pairwiseLTR", pw,
                   lambda fn, cp, c, w, z: fn(cp, c))

    # 13. Pointwise Learning-to-Rank
    pt = pyepo.func.pointwiseLTR(optmodel, processes=1, solve_ratio=1, dataset=dataset)
    train_and_eval("pointwiseLTR", pt,
                   lambda fn, cp, c, w, z: fn(cp, c))

    # ============================================================
    # Part C: Training with Pyomo backend (SPO+ only)
    # ============================================================
    try:
        from pyepo.model.omo.knapsack import knapsackModel as omoKnapsack
        print("\n" + "="*60)
        print("Part C: Training with Pyomo Backend (SPO+)")
        print("="*60)

        omo_optmodel = omoKnapsack(weights, capacity, solver="gurobi")
        omo_dataset = pyepo.data.dataset.optDataset(omo_optmodel, x, c)
        omo_dataloader = DataLoader(omo_dataset, batch_size=32, shuffle=True)

        pred_omo = LinearRegression(num_feat, num_item)
        opt_omo = torch.optim.Adam(pred_omo.parameters(), lr=1e-2)
        spo_omo = pyepo.func.SPOPlus(omo_optmodel, processes=2)
        print("\n--- SPOPlus (Pyomo/Gurobi) ---")
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            for data in omo_dataloader:
                xi, ci, wi, zi = data
                cp = pred_omo(xi)
                loss = spo_omo(cp, ci, wi, zi)
                opt_omo.zero_grad()
                loss.backward()
                opt_omo.step()
                epoch_loss += loss.item()
                num_batches += 1
            print("  Epoch {:3d}/{}: avg loss = {:.4f}".format(
                epoch + 1, num_epochs, epoch_loss / num_batches))
        reg = pyepo.metric.regret(pred_omo, omo_optmodel, omo_dataloader)
        print("  Regret: {:.4f}".format(reg))
    except ImportError as e:
        print("\n[SKIP] Pyomo not available: {}".format(e))

    # ============================================================
    # Part D: Training with COPT backend (SPO+ only)
    # ============================================================
    try:
        from pyepo.model.copt.knapsack import knapsackModel as coptKnapsack
        print("\n" + "="*60)
        print("Part D: Training with COPT Backend (SPO+)")
        print("="*60)

        copt_optmodel = coptKnapsack(weights, capacity)
        copt_dataset = pyepo.data.dataset.optDataset(copt_optmodel, x, c)
        copt_dataloader = DataLoader(copt_dataset, batch_size=32, shuffle=True)

        pred_copt = LinearRegression(num_feat, num_item)
        opt_copt = torch.optim.Adam(pred_copt.parameters(), lr=1e-2)
        spo_copt = pyepo.func.SPOPlus(copt_optmodel, processes=2)
        print("\n--- SPOPlus (COPT) ---")
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            for data in copt_dataloader:
                xi, ci, wi, zi = data
                cp = pred_copt(xi)
                loss = spo_copt(cp, ci, wi, zi)
                opt_copt.zero_grad()
                loss.backward()
                opt_copt.step()
                epoch_loss += loss.item()
                num_batches += 1
            print("  Epoch {:3d}/{}: avg loss = {:.4f}".format(
                epoch + 1, num_epochs, epoch_loss / num_batches))
        reg = pyepo.metric.regret(pred_copt, copt_optmodel, copt_dataloader)
        print("  Regret: {:.4f}".format(reg))
    except ImportError as e:
        print("\n[SKIP] COPT not available: {}".format(e))

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
