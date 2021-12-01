#!/usr/bin/env python
# coding: utf-8
"""
Model training
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import torch
from torch.utils.data import DataLoader

import spo
from run import net, utils


def train(trainset, testset, model, config):
    """
    training for preditc-then-optimize
    """
    print("Training...")
    if config.mthd == "2s":
        print("Using auto two-stage predict then optimize...")
        res = train2Stage(trainset, model, config)
    if config.mthd == "spo":
        print("Using SPO+ loss...")
        trainloader = DataLoader(trainset, batch_size=config.batch, shuffle=True)
        testloader = DataLoader(testset, batch_size=config.batch, shuffle=False)
        res = trainSPO(trainloader, testloader, model, config)
    if config.mthd == "bb":
        print("Using Black-box optimizer block...")
        trainloader = DataLoader(trainset, batch_size=config.batch, shuffle=True)
        testloader = DataLoader(testset, batch_size=config.batch, shuffle=False)
        res = trainBB(trainloader, testloader, model, config)
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
        arch.append(config.items)
    if config.prob == "tsp":
        arch.append(config.nodes * (config.nodes - 1) // 2)
    reg = net.fcNet(arch)
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
    # prediction model
    if config.pred == "lr":
        predictor = LinearRegression()
        twostage = spo.twostage.sklearnPred(predictor)
    if config.pred == "rf":
        predictor = RandomForestRegressor(random_state=config.seed)
        twostage = spo.twostage.sklearnPred(predictor)
    if config.pred == "auto":
        print("Running with Auto-SKlearn...")
        twostage = spo.twostage.autoSklearnPred(model, config.seed)
    # training
    twostage.fit(trainset.x, trainset.c)
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
    spo.train.trainSPO(reg, model, optimizer, trainloader, testloader,
                       logdir=logdir, epoch=config.epoch, processes=config.proc,
                       l1_lambd=config.l1, l2_lambd=config.l2, log=config.elog)
    return reg


def trainBB(trainloader, testloader, model, config):
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
    spo.train.trainBB(reg, model, optimizer, trainloader, testloader,
                      logdir=logdir, epoch=config.epoch, processes=config.proc,
                      bb_lambd=config.smth, l1_lambd=config.l1,
                      l2_lambd=config.l2, log=config.elog)
    return reg
