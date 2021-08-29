#!/usr/bin/env python
# coding: utf-8
"""
SPO training pipeline
"""
import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import torch
from torch.utils.data import DataLoader

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
        print("Experiment {}:".format(i))
        # generate data
        data = genData(config)
        if config.problem == "ks":
            config.wghts, data = data[0], (data[1], data[2])
        # build model
        model = buildModel(config)
        # relax model
        if config.relax:
            print("Building model relaxation...")
            model_rel = model.relax()
        # build data loader
        dataset_train, dataset_test, loader_train, loader_test = \
        buildDataLoader(data, model, config)
        # training
        if config.mthd == "2s":
            print("Using two-stage predict then optimize...")
        if config.mthd == "spo":
            print("Using SPO+ loss...")
        if config.mthd == "bb":
            print("Using Black-box optimzer block...")
        print()


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
        model_rel = model.relax()
        dataset_train = spo.data.dataset.optDataset(model_rel, x_train, c_train)
    else:
        dataset_train = spo.data.dataset.optDataset(model, x_train, c_train)
    dataset_test = spo.data.dataset.optDataset(model, x_test, c_test)
    # get data loader
    print("Building Pytorch DataLoader...")
    loader_train = DataLoader(dataset_train,
                              batch_size=config.batch,
                              shuffle=True)
    loader_test = DataLoader(dataset_test,
                             batch_size=config.batch,
                             shuffle=False)
    return dataset_train, dataset_test, loader_train, loader_test


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
                        help="predictor of two-stage predict then optimiza")

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

    # get configuration
    config = parser.parse_args()

    # run experiment pipeline
    pipeline(config)
