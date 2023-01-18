#!/usr/bin/env python
# coding: utf-8
"""
Model training
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import torch
from torch.utils.data import DataLoader

import pyepo
from run import net, utils, training


def train(trainset, testset, model, config):
    """
    training for preditc-then-optimize
    """
    print("Training...")
    if config.mthd == "2s":
        print("Using two-stage predict then optimize...")
        res = train2Stage(trainset, model, config)
    if config.mthd == "spo":
        print("Using SPO+ loss...")
        trainloader = DataLoader(trainset, batch_size=config.batch, shuffle=True)
        testloader = DataLoader(testset, batch_size=config.batch, shuffle=False)
        res = trainSPO(trainloader, testloader, model, config)
    if config.mthd == "dbb":
        print("Using Black-box optimizer block...")
        trainloader = DataLoader(trainset, batch_size=config.batch, shuffle=True)
        testloader = DataLoader(testset, batch_size=config.batch, shuffle=False)
        res = trainDBB(trainloader, testloader, model, config)
    if config.mthd == "dpo":
        print("Using differentiable perturbed optimizer...")
        trainloader = DataLoader(trainset, batch_size=config.batch, shuffle=True)
        testloader = DataLoader(testset, batch_size=config.batch, shuffle=False)
        res = trainDPO(trainloader, testloader, model, config)
    if config.mthd == "pfyl":
        print("Using perturbed Fenchel-Young loss...")
        trainloader = DataLoader(trainset, batch_size=config.batch, shuffle=True)
        testloader = DataLoader(testset, batch_size=config.batch, shuffle=False)
        res = trainPFYL(trainloader, testloader, model, config)
    return res


def trainInit(config):
    """
    initiate neural network and optimizer
    """
    # build nn
    arch = [config.feat] + config.net
    if config.prob == "sp":
        arch.append((config.grid[0] - 1) * config.grid[1] + \
                    (config.grid[1] - 1) * config.grid[0])
    if config.prob == "ks":
        arch.append(config.item)
    if config.prob == "tsp":
        arch.append(config.nodes * (config.nodes - 1) // 2)
    reg = net.fcNet(arch, softplus=config.sftp)
    # set optimizer
    if config.optm == "sgd":
        optimizer = torch.optim.SGD(reg.parameters(), lr=config.lr)
    if config.optm == "adam":
        optimizer = torch.optim.Adam(reg.parameters(), lr=config.lr)
    return reg, optimizer


def train2Stage(trainset, model, config):
    """
    two-stage preditc-then-optimize training
    """
    feats, costs = trainset.feats, trainset.costs
    # prediction model
    if config.pred == "lr":
        predictor = LinearRegression()
        twostage = pyepo.twostage.sklearnPred(predictor)
    if config.pred == "rf":
        predictor = RandomForestRegressor(random_state=config.seed)
        twostage = pyepo.twostage.sklearnPred(predictor)
    if config.pred == "auto":
        print("Running with Auto-SKlearn...")
        if config.metric == "mse":
            twostage = pyepo.twostage.autoSklearnPred(model, config.seed,
                                                      timelimit=600,
                                                      metric=config.metric)
        if config.metric == "regret":
            twostage = pyepo.twostage.autoSklearnPred(model, config.seed,
                                                      timelimit=3000,
                                                      metric=config.metric)
        # avoid to be multiclass
        if config.prob == "ks":
            costs += np.random.randn(*costs.shape) * 1e-5
    # training
    twostage.fit(feats, costs)
    return twostage


def trainSPO(trainloader, testloader, model, config):
    """
    SPO+ training
    """
    # init
    reg, optimizer = trainInit(config)
    # relax
    if config.rel:
        model = model.relax()
    # log dir
    logdir = "./logs" + utils.getSavePath(config)[5:-4]
    # train
    training.trainSPO(reg, model, optimizer, trainloader, testloader,
                      logdir=logdir, epoch=config.epoch,
                      processes=config.proc, l1_lambd=config.l1,
                      l2_lambd=config.l2, log=config.elog)
    return reg


def trainDBB(trainloader, testloader, model, config):
    """
    Black-Box training
    """
    # init
    reg, optimizer = trainInit(config)
    # relax
    if config.rel:
        model = model.relax()
    # log dir
    logdir = "./logs" + utils.getSavePath(config)[5:-4]
    # train
    training.trainDBB(reg, model, optimizer, trainloader, testloader,
                     lossfunc=config.loss, logdir=logdir, epoch=config.epoch,
                     processes=config.proc, bb_lambd=config.smth,
                     l1_lambd=config.l1, l2_lambd=config.l2, log=config.elog)
    return reg


def trainPFYL(trainloader, testloader, model, config):
    """
    Fenchel-Young loss training
    """
    # init
    reg, optimizer = trainInit(config)
    # relax
    if config.rel:
        model = model.relax()
    # log dir
    logdir = "./logs" + utils.getSavePath(config)[5:-4]
    # train
    training.trainPFYL(reg, model, optimizer, trainloader, testloader,
                       logdir=logdir, epoch=config.epoch, processes=config.proc,
                       n_samples=config.samp, epsilon=config.eps, l1_lambd=config.l1,
                       l2_lambd=config.l2, log=config.elog)
    return reg
