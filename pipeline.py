#!/usr/bin/env python
# coding: utf-8
"""
SPO training pipeline
"""

import os
import time
import random

import numpy as np
import pandas as pd
import torch

import pyepo
from run import utils
from run import train
from run import eval

def pipeline(config):
    # shortest path
    if config.prob == "sp":
        print("Running experiments for shortest path:")
    # knapsack
    if config.prob == "ks":
        print("Running experiments for multi-dimensional knapsack:")
    # travelling salesman
    if config.prob == "tsp":
        print("Running experiments for traveling salesman:")
    print()
    # create table
    save_path = utils.getSavePath(config)
    print(save_path)
    if os.path.isfile(save_path): # exist res
        df = pd.read_csv(save_path)
        skip = True # skip flag
    else:
        df = pd.DataFrame(columns=["True SPO", "Unamb SPO", "MSE", "Elapsed", "Epochs"])
        skip = False # skip flag

    for i in range(config.expnum):
        # random seed for each experiment
        config.seed = i
        # set random seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        # start exp
        print("===============================================================")
        print("Experiment {}:".format(i))
        print("===============================================================")
        # generate data
        data = utils.genData(config)
        if config.prob == "ks":
            config.wght, data = data[0], (data[1], data[2])
        print()
        # skip exist experiments
        if skip and (i < len(df)):
            print("Skip experiment {}.".format(i))
            print()
            continue
        else:
            skip = False
        # build model
        model = utils.buildModel(config)
        # build data loader
        trainset, testset = utils.buildDataSet(data, model, config)
        print()
        # train
        tick = time.time()
        res = train(trainset, testset, model, config)
        tock = time.time()
        elapsed = tock - tick
        print("Time elapsed: {:.4f} sec".format(elapsed))
        print()
        # evaluate
        truespo, unambspo, mse = eval(testset, res, model, config)
        # save
        epoch = 0 if config.mthd == "2s" else config.epoch
        row = {"True SPO":truespo, "Unamb SPO":unambspo, "MSE":mse,
               "Elapsed":elapsed, "Epochs":epoch}
        df = df.append(row, ignore_index=True)
        df.to_csv(save_path, index=False)
        # autosklean model info
        if config.mthd == "2s" and config.pred == "auto":
            cv_result = pd.DataFrame.from_dict(res.cv_results_)
            cv_result["Experiment"] = i
            if i == 0:
                df_cv = cv_result
            else:
                df_cv = pd.read_csv(save_path[:-4]+"-cv.csv")
                df_cv = df_cv.append(cv_result, ignore_index=True)
            df_cv.to_csv(save_path[:-4]+"-cv.csv", index=False)
        print("Saved to " + save_path + ".")
        print("\n\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # experiments configuration
    parser.add_argument("--mthd",
                        type=str,
                        default="spo",
                        choices=["2s", "spo", "dbb", "dpo", "pfyl"],
                        help="method")
    parser.add_argument("--expnum",
                        type=int,
                        default=1,
                        help="number of experiments")
    parser.add_argument("--loss",
                        type=str,
                        default="r",
                        choices=["r", "h"],
                        help="loss function for Black-Box")
    parser.add_argument("--rel",
                        action="store_true",
                        help="train with relaxation model")
    parser.add_argument("--pred",
                        type=str,
                        default="lr",
                        choices=["auto", "lr", "rf"],
                        help="predictor of two-stage predict then optimize")
    parser.add_argument("--metric",
                        type=str,
                        default="mse",
                        choices=["regret", "mse"],
                        help="metric for auto-sklearm predictor")
    parser.add_argument("--elog",
                        type=int,
                        default=0,
                        help="steps of evluation and log")
    parser.add_argument("--form",
                        type=str,
                        default="gg",
                        choices=["gg", "dfj", "mtz"],
                        help="TSP formulation")
    parser.add_argument("--path",
                        type=str,
                        default="./res",
                        help="path to save result")

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
                        default=1,
                        help="features polynomial degree")
    parser.add_argument("--noise",
                        type=float,
                        default=0,
                        help="noise half-width")

    # optimization model configuration
    parser.add_argument("--prob",
                        type=str,
                        default="sp",
                        choices=["sp", "ks", "tsp"],
                        help="problem type")
    # shortest path
    parser.add_argument("--grid",
                        type=int,
                        nargs=2,
                        default=(5,5),
                        help="network grid for shortest path")
    # knapsack
    parser.add_argument("--item",
                        type=int,
                        default=48,
                        help="number of items for knapsack")
    parser.add_argument("--dim",
                        type=int,
                        default=3,
                        help="dimension for knapsack")
    parser.add_argument("--cap",
                        type=int,
                        default=30,
                        help="capacity for knapsack")
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
    parser.add_argument("--sftp",
                        action="store_true",
                        help="positive prediction with SoftPlus activation")
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
                        default=0.0,
                        help="l1 regularization parameter")
    parser.add_argument("--l2",
                        type=float,
                        default=0.0,
                        help="l2 regularization parameter")
    parser.add_argument("--smth",
                        type=int,
                        default=10,
                        help="smoothing parameter for Black-Box")
    parser.add_argument("--samp",
                        type=int,
                        default=1,
                        help="number of samples for perturbed methods")
    parser.add_argument("--sig",
                        type=float,
                        default=1.0,
                        help="amplitude parameter for perturbed methods")
    parser.add_argument("--proc",
                        type=int,
                        default=1,
                        help="number of processor for optimization")

    # get configuration
    config = parser.parse_args()

    # run experiment pipeline
    pipeline(config)

# python pipeline.py --prob sp --mthd spo --lan gurobi --data 1000 --deg 1 --noise 0 --epoch 100 --lr 1e-2 --proc 1
# tensorboard --logdir logs
