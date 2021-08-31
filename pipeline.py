#!/usr/bin/env python
# coding: utf-8
"""
SPO training pipeline
"""
import argparse
import os
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import spo

def pipeline(config):
    # shortest path
    if config.problem == "sp":
        print("Running experiments for shortest path problem:")
    # knapsack
    if config.problem == "ks":
        print("Running experiments for multi-dimensional knapsack problem:")
    # travelling salesman
    if config.problem == "tsp":
        print("Running experiments for traveling salesman problem:")
    print()

    # set random seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    for i in range(config.expnum):
        config.seed = np.random.randint(999)
        print("===============================================================")
        print("Experiment {}:".format(i))
        print("===============================================================")
        # generate data
        data = genData(config)
        if config.problem == "ks":
            config.wghts, data = data[0], (data[1], data[2])
        print()
        # build model
        model = buildModel(config)
        # build data loader
        trainset, testset, trainloader, testloader = buildDataLoader(data, model,
                                                                     config)
        print()
        # train
        res = train(trainset, testset, trainloader, testloader, model, config)
        print()
        # evaluate
        truespo, unambspo = eval(testset, testloader, res, model, config)
        print("\n\n")


def genData(config):
    """
    generate synthetic data
    """
    print("Generating synthetic data...")
    # shortest path
    if config.problem == "sp":
        data = spo.data.shortestpath.genData(config.data+1000, config.feat,
                                             config.grid, deg=config.deg,
                                             noise_width=config.noise,
                                             seed=config.seed)
    # knapsack
    if config.problem == "ks":
        data = spo.data.knapsack.genData(config.data+1000, config.feat,
                                         config.items,  dim=config.dim,
                                         deg=config.deg,
                                         noise_width=config.noise,
                                         seed=config.seed)
    # travelling salesman
    if config.problem == "tsp":
        data = spo.data.tsp.genData(config.data+1000, config.feat, config.nodes,
                                    deg=config.deg, noise_width=config.noise,
                                    seed=config.seed)
    return data


def buildModel(config):
    """
    build optimization model
    """
    # shortest path
    if config.problem == "sp":
        if config.lan == "gurobi":
            print("Building model with GurobiPy...")
            model = spo.model.grb.shortestPathModel(config.grid)
        if config.lan == "pyomo":
            print("Building model with Pyomo...")
            model = spo.model.omo.shortestPathModel(config.grid, config.solver)
    # knapsack
    if config.problem == "ks":
        if len(config.caps) != len(config.wghts):
            raise ValueError("Dimensional inconsistency: {} caps, {} wghts".
                              format(len(config.caps), len(config.wghts)))
        if config.lan == "gurobi":
            print("Building model with GurobiPy...")
            model = spo.model.grb.knapsackModel(config.wghts, config.caps)
        if config.lan == "pyomo":
            print("Building model with Pyomo...")
            model = spo.model.omo.knapsackModel(config.wghts, config.caps,
                                                config.solver)
    # travelling salesman
    if config.problem == "tsp":
        if config.lan == "gurobi":
            print("Building model with GurobiPy...")
            model = spo.model.grb.tspGGModel(config.nodes)
        if config.lan == "pyomo":
            raise RuntimeError("TSP with Pyomo is not implemented.")
    return model


def buildDataLoader(data, model, config):
    """
    build Pytorch DataLoader
    """
    x, c = data
    # data split
    x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=1000,
                                                        random_state=config.seed)
    # build data set
    if config.relax:
        print("Building relaxation model...")
        model_rel = model.relax()
        trainset = spo.data.dataset.optDataset(model_rel, x_train, c_train)
    else:
        trainset = spo.data.dataset.optDataset(model, x_train, c_train)
    testset = spo.data.dataset.optDataset(model, x_test, c_test)
    # get data loader
    print("Building Pytorch DataLoader...")
    trainloader = DataLoader(trainset, batch_size=config.batch, shuffle=True)
    testloader = DataLoader(testset, batch_size=config.batch, shuffle=False)
    return trainset, testset, trainloader, testloader


def train(trainset, testset, trainloader, testloader, model, config):
    """
    training for preditc-then-optimize
    """
    print("Training...")
    if config.mthd == "2s":
        print("Using two-stage predict then optimize...")
        res = train2Stage(trainset, model, config)
    if config.mthd == "spo":
        print("Using SPO+ loss...")
        res = trainSPO(trainset, testset, trainloader, testloader, model, config)
    if config.mthd == "bb":
        print("Using Black-box optimzer block...")
        res = None
    return res


def train2Stage(trainset, model, config):
    """
    two-stage preditc-then-optimize training
    """
    # prediction model
    if config.pred == "lr":
        predictor = LinearRegression()
    if config.pred == "rf":
        predictor = RandomForestRegressor(random_state=config.seed)
    # two-stage model
    if config.relax:
        print("Building relaxation model...")
        model_rel = model.relax()
        twostage = spo.twostage.sklearnPred(predictor, model_rel)
    else:
        twostage = spo.twostage.sklearnPred(predictor, model)
    # training
    twostage.fit(trainset.x, trainset.c)
    return twostage

def trainSPO(trainset, testset, trainloader, testloader, model, config):
    """
    SPO+ training
    """
    # init
    reg, optimizer, device = trainInit(config)
    # training mode
    reg.train()
    # set SPO+ Loss as criterion
    criterion = spo.func.SPOPlus(model, processes=config.proc)
    # train
    time.sleep(1)
    for epoch in tqdm(range(config.epoch)):
        # load data
        for i, data in enumerate(trainloader):
            x, c, w, z = data
            x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
            # forward pass
            cp = reg(x)
            loss = criterion.apply(cp, c, w, z).mean()
            # l1 reg
            if config.l1:
                l1_reg = torch.abs(cp - c).sum(dim=1).mean()
                loss += config.l1 * l1_reg
            # l2 reg
            if config.l2:
                l2_reg = ((cp - c) ** 2).sum(dim=1).mean()
                loss += config.l2 * l2_reg
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return reg

def trainInit(config):
    """
    initiate neural network and optimizer
    """
    # build nn
    arch = [config.feat] + config.net
    if config.problem == "sp":
        arch.append((config.grid[0] - 1) * config.grid[1] + \
                    (config.grid[1] - 1) * config.grid[0])
    if config.problem == "ks":
        arch.append(config.items)
    if config.problem == "tsp":
        arch.append(config.nodes * (config.nodes - 1) // 2)
    reg = fcNet(arch)
    # set optimzer
    if config.optm == "sgd":
        optimizer = torch.optim.SGD(reg.parameters(), lr=config.lr)
    if config.optm == "adam":
        optimizer = torch.optim.Adam(reg.parameters(), lr=config.lr)
    # get device
    device = getDevice()
    reg.to(device)
    return reg, optimizer, device


def getDevice():
    """
    get device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print("    {}:".format(i), torch.cuda.get_device_name(i))
    else:
        device = torch.device("cpu")
        print("Device: CPU")
    return device


def eval(testset, testloader, res, model, config):
    """
    evaluate permance
    """
    print("Evaluating...")
    if config.mthd == "2s":
        # prediction
        c_test_pred = res.predict(testset.x)
        truespo = 0
        unambspo = 0
        for i in tqdm(range(len(testset))):
            cp_i = c_test_pred[i]
            c_i = testset.c[i]
            z_i = testset.z[i,0]
            truespo += spo.eval.calTrueSPO(model, cp_i, c_i, z_i)
            unambspo += spo.eval.calUnambSPO(model, cp_i, c_i, z_i)
        truespo /= abs(testset.z.sum())
        unambspo /= abs(testset.z.sum())
        time.sleep(1)
    if (config.mthd == "spo") or (config.mthd == "bb"):
        truespo = spo.eval.trueSPO(res, model, testloader)
        unambspo = spo.eval.unambSPO(res, model, testloader)
    print('Normalized true SPO Loss: {:.2f}%'.format(truespo * 100))
    print('Normalized unambiguous SPO Loss: {:.2f}%'.format(unambspo * 100))
    return truespo, unambspo


class fcNet(nn.Module):
    """
    multi-layer fully connected neural network regression
    """
    def __init__(self, arch):
        super().__init__()
        layers = []
        for i in range(len(arch)-1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # experiments configuration
    parser.add_argument("--mthd",
                        type=str,
                        default="spo",
                        choices=["2s", "spo", "bb"],
                        help="method")
    parser.add_argument("--seed",
                        type=int,
                        default=135,
                        help="random seed")
    parser.add_argument("--expnum",
                        type=int,
                        default=10,
                        help="number of experiments")
    parser.add_argument("--relax",
                        action="store_true",
                        help="train with relaxation model")
    parser.add_argument("--pred",
                        type=str,
                        default="lr",
                        choices=["lr", "rf"],
                        help="predictor of two-stage predict then optimize")

    # solver configuration
    parser.add_argument("--lan",
                        type=str,
                        default="gurobi",
                        choices=["gurobi", "pyomo"],
                        help="modeling language")
    parser.add_argument("--solver",
                        type=str,
                        default="gurobi",
                        help="solver for Pyomo")

    # data configuration
    parser.add_argument("--data",
                        type=int,
                        default=1000,
                        help="training data size")
    parser.add_argument("--feat",
                        type=int,
                        default=5,
                        help="feature size")
    parser.add_argument("--deg",
                        type=int,
                        default=4,
                        help="features polynomial degree")
    parser.add_argument("--noise",
                        type=float,
                        default=0,
                        help="noise half-width")

    # optimization model configuration
    parser.add_argument("--problem",
                        type=str,
                        default="sp",
                        choices=["sp", "ks", "tsp"],
                        help="problem type")
    # shortest path
    parser.add_argument("--grid",
                        type=int,
                        nargs=2,
                        default=(20,20),
                        help="network grid for shortest path")
    # knapsack
    parser.add_argument("--items",
                        type=int,
                        default=48,
                        help="number of items for knapsack")
    parser.add_argument("--dim",
                        type=int,
                        default=3,
                        help="dimension for knapsack")
    parser.add_argument("--caps",
                        type=float,
                        nargs='+',
                        default=(30,30,30),
                        help="dimension for knapsack")
    # tsp
    parser.add_argument("--nodes",
                        type=int,
                        default=20,
                        help="number of nodes")

    # training configuration
    parser.add_argument("--batch",
                        type=int,
                        default=32,
                        help="batch size")
    parser.add_argument("--epoch",
                        type=int,
                        default=100,
                        help="number of epochs")
    parser.add_argument("--net",
                        type=int,
                        nargs='*',
                        default=[],
                        help="size of neural network hidden layers")
    parser.add_argument("--optm",
                        type=str,
                        default="adam",
                        choices=["sgd", "adam"],
                        help="optimizer neural network")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-3,
                        help="learning rate")
    parser.add_argument("--l1",
                        type=float,
                        default=0,
                        help="l1 regularization parameter")
    parser.add_argument("--l2",
                        type=float,
                        default=0,
                        help="l2 regularization parameter")
    parser.add_argument("--proc",
                        type=int,
                        default=1,
                        help="number of processor for optimization")


    # get configuration
    config = parser.parse_args()

    # run experiment pipeline
    pipeline(config)
