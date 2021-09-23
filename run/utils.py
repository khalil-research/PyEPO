#!/usr/bin/env python
# coding: utf-8
"""
Utilities
"""
import os

from sklearn.model_selection import train_test_split

import spo

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
        path += "/" + "i{}d{}c{}".format(config.items, config.dim, config.cap)
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
    # method
    filename = config.mthd
    if config.mthd == "2s":
        filename += "-" + config.pred
    else:
        if not config.net:
            filename += "-lr"
        else:
            filename += "-fc" +"-".join(config.net)
    if config.mthd == "bb":
        filename += "-λ{}".format(config.smth)
    if config.rel:
        filename += "-rel"
    # data size
    filename += "_n{}p{}".format(config.data, config.feat)
    # degree
    filename += "_d{}".format(config.deg)
    # noise
    filename += "_e{}".format(config.noise)
    # optimizer
    filename += "_" + config.optm + str(config.lr)
    # regularization
    filename += "_l1{}l2{}".format(config.l1, config.l2)
    # processors
    filename += "_c{}".format(config.proc)
    return path + "/" + filename + ".csv"


def genData(config):
    """
    generate synthetic data
    """
    print("Generating synthetic data...")
    # shortest path
    if config.prob == "sp":
        data = spo.data.shortestpath.genData(config.data+1000, config.feat,
                                             config.grid, deg=config.deg,
                                             noise_width=config.noise,
                                             seed=config.seed)
    # knapsack
    if config.prob == "ks":
        data = spo.data.knapsack.genData(config.data+1000, config.feat,
                                         config.items,  dim=config.dim,
                                         deg=config.deg,
                                         noise_width=config.noise,
                                         seed=config.seed)
    # travelling salesman
    if config.prob == "tsp":
        data = spo.data.tsp.genData(config.data+1000, config.feat, config.nodes,
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
            model = spo.model.grb.shortestPathModel(config.grid)
        if config.lan == "pyomo":
            print("Building model with Pyomo...")
            model = spo.model.omo.shortestPathModel(config.grid, config.solver)
    # knapsack
    if config.prob == "ks":
        caps = [config.cap] * config.dim
        if config.lan == "gurobi":
            print("Building model with GurobiPy...")
            model = spo.model.grb.knapsackModel(config.wght, caps)
        if config.lan == "pyomo":
            print("Building model with Pyomo...")
            model = spo.model.omo.knapsackModel(config.wght, caps, config.solver)
    # travelling salesman
    if config.prob == "tsp":
        if config.lan == "gurobi":
            print("Building model with GurobiPy...")
            if config.form == "gg":
                print("Using Gavish–Graves formulation...")
                model = spo.model.grb.tspGGModel(config.nodes)
            if config.form == "dfj":
                print("Using Danzig–Fulkerson–Johnson formulation...")
                model = spo.model.grb.tspDFJModel(config.nodes)
            if config.form == "mtz":
                print("Using Miller-Tucker-Zemlin formulation...")
                model = spo.model.grb.tspMTZModel(config.nodes)
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
        trainset = spo.data.dataset.optDataset(model_rel, x_train, c_train)
    else:
        trainset = spo.data.dataset.optDataset(model, x_train, c_train)
    testset = spo.data.dataset.optDataset(model, x_test, c_test)
    return trainset, testset
