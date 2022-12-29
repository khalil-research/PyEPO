#!/usr/bin/env python
# coding: utf-8
"""
Utilities
"""
import os

from sklearn.model_selection import train_test_split

import pyepo

def getSavePath(config):
    """
    get file path to save result
    """
    path = config.path
    if not os.path.isdir(path):
        os.mkdir(path)
    # problem type
    path += "/" + config.prob
    if not os.path.isdir(path):
        os.mkdir(path)
    # problem specification
    if config.prob == "sp":
        path += "/" + "h{}w{}".format(*config.grid)
    if config.prob == "ks":
        path += "/" + "i{}d{}c{}".format(config.item, config.dim, config.cap)
    if config.prob == "tsp":
        path += "/" + "n{}".format(config.nodes)
    if not os.path.isdir(path):
        os.mkdir(path)
    # formulation
    if config.prob == "tsp":
        path += "/" + config.form
        if not os.path.isdir(path):
            os.mkdir(path)
    # solver
    path += "/" + config.lan
    if config.lan == "pyomo":
        path += "-" + config.solver
    if not os.path.isdir(path):
        os.mkdir(path)
    # data size
    filename = "n{}p{}".format(config.data, config.feat)
    # degree
    filename += "-d{}".format(config.deg)
    # noise
    filename += "-e{}".format(config.noise)
    # method
    filename += "_" + config.mthd
    if config.mthd == "2s":
        filename += "-" + config.pred
    if config.rel:
       filename += "-rel"
    if config.mthd != "2s":
        if not config.net:
            filename += "_lr"
        else:
            filename += "_fc" +"-".join(map(str, config.net))
        # optimizer
        filename += "_" + config.optm + str(config.lr)
        # batch size
        filename += "_bs{}".format(config.batch)
        # regularization
        filename += "_l1{}l2{}".format(config.l1, config.l2)
        # processors
        filename += "_c{}".format(config.proc)
    if config.mthd == "dbb":
        filename += "-lamb{}".format(config.smth)
        if config.loss == "h":
            filename += "-h"
    if config.mthd == "2s" and config.pred == "auto":
        if config.metric == "mse":
            filename += "-mse"
    if config.mthd != "2s":
        # softplus
        if config.sftp:
            filename += "-sf"
    return path + "/" + filename + ".csv"


def genData(config):
    """
    generate synthetic data
    """
    print("Generating synthetic data...")
    # shortest path
    if config.prob == "sp":
        data = pyepo.data.shortestpath.genData(config.data+1000, config.feat,
                                               config.grid, deg=config.deg,
                                               noise_width=config.noise,
                                               seed=config.seed)
    # knapsack
    if config.prob == "ks":
        data = pyepo.data.knapsack.genData(config.data+1000, config.feat,
                                           config.item, dim=config.dim,
                                           deg=config.deg,
                                           noise_width=config.noise,
                                           seed=config.seed)
    # travelling salesman
    if config.prob == "tsp":
        data = pyepo.data.tsp.genData(config.data+1000, config.feat, config.nodes,
                                      deg=config.deg, noise_width=config.noise,
                                      seed=config.seed)
    return data


def buildModel(config):
    """
    build optimization model
    """
    # shortest path
    if config.prob == "sp":
        if config.lan == "gurobi":
            print("Building model with GurobiPy...")
            model = pyepo.model.grb.shortestPathModel(config.grid)
        if config.lan == "pyomo":
            print("Building model with Pyomo...")
            model = pyepo.model.omo.shortestPathModel(config.grid, config.solver)
    # knapsack
    if config.prob == "ks":
        caps = [config.cap] * config.dim
        if config.lan == "gurobi":
            print("Building model with GurobiPy...")
            model = pyepo.model.grb.knapsackModel(config.wght, caps)
        if config.lan == "pyomo":
            print("Building model with Pyomo...")
            model = pyepo.model.omo.knapsackModel(config.wght, caps, config.solver)
    # travelling salesman
    if config.prob == "tsp":
        if config.lan == "gurobi":
            print("Building model with GurobiPy...")
            if config.form == "gg":
                print("Using Gavish–Graves formulation...")
                model = pyepo.model.grb.tspGGModel(config.nodes)
            if config.form == "dfj":
                print("Using Danzig–Fulkerson–Johnson formulation...")
                model = pyepo.model.grb.tspDFJModel(config.nodes)
            if config.form == "mtz":
                print("Using Miller-Tucker-Zemlin formulation...")
                model = pyepo.model.grb.tspMTZModel(config.nodes)
        if config.lan == "pyomo":
            raise RuntimeError("TSP with Pyomo is not implemented.")
    return model


def buildDataSet(data, model, config):
    """
    build Pytorch DataSet
    """
    x, c = data
    # data split
    x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=1000,
                                                        random_state=config.seed)
    # build data set
    if config.rel:
        print("Building relaxation model...")
        model_rel = model.relax()
        trainset = pyepo.data.dataset.optDataset(model_rel, x_train, c_train)
    else:
        trainset = pyepo.data.dataset.optDataset(model, x_train, c_train)
    testset = pyepo.data.dataset.optDataset(model, x_test, c_test)
    return trainset, testset
