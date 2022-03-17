#!/usr/bin/env python
# coding: utf-8
"""
Knapsack configuration
"""

from copy import deepcopy
from types import SimpleNamespace

# init configs
config = SimpleNamespace()
configKS = {}


### ========================= general setting =========================
## optimization model configuration
# problem type
config.prob = "ks"
# number of items
config.item = 32
# dimension
config.dim = None
# capacity
config.cap = 20

## experiments configuration
# number of experiments
config.expnum = None
# seed
config.seed = 135
# relaxation
config.rel = False
# steps of evluation and log
config.elog = 0
# path to save result
config.path="./res"

## solver configuration
# modelling language
config.lan = "gurobi"
# solver for Pyomo
#config.solver = "gurobi"

## data configuration
# training data size
config.data = None
# feature size
config.feat = 5
# features polynomial degree
config.deg = None
# noise half-width
config.noise = None


### ========================= lr =========================
config_lr = deepcopy(config)

## experiments configuration
# method
config_lr.mthd = "2s"
# predictor
config_lr.pred = "lr"
# time limit
config_lr.timeout = 3

configKS["lr"] = config_lr

### ========================= rf =========================
config_rf = deepcopy(config)

## experiments configuration
# method
config_rf.mthd = "2s"
# predictor
config_rf.pred = "rf"
# time limit
config_rf.timeout = 6

configKS["rf"] = config_rf

### ========================= auto =========================
config_auto = deepcopy(config)

## experiments configuration
# method
config_auto.mthd = "2s"
# predictor
config_auto.pred = "auto"
# metric
config_auto.metric = "mse"
# time limit
config_auto.timeout = 18

configKS["auto"] = config_auto

### ========================= SPO =========================
config_spo = deepcopy(config)

## experiments configuration
# method
config_spo.mthd = "spo"
# time limit
config_spo.timeout = 30

## training configuration
# size of neural network hidden layers
config_spo.net = []
# number of epochs
config_spo.batch = 32
# number of epochs
config_spo.epoch = None
# optimizer neural network
config_spo.optm = "adam"
# learning rate
config_spo.lr = 1e-2
# l1 regularization parameter
config_spo.l1 = 0.0
# l2 regularization parameter
config_spo.l2 = 0.0
# number of processor for optimization
config_spo.proc = 32

configKS["spo"] = config_spo


### ======================== DBB =========================
config_bb = deepcopy(config)

## experiments configuration
# method
config_bb.mthd = "bb"
# loss
config_bb.loss = "r"
# time limit
config_bb.timeout = 45

## training configuration
# size of neural network hidden layers
config_bb.net = []
# number of epochs
config_bb.batch = 32
# number of epochs
config_bb.epoch = None
# optimizer neural network
config_bb.optm = "adam"
# learning rate
config_bb.lr = 5e-2
# smoothing parameter for Black-Box
config_bb.smth = 10
# l1 regularization parameter
config_bb.l1 = 0.0
# l2 regularization parameter
config_bb.l2 = 0.0
# number of processor for optimization
config_bb.proc = 32

configKS["bb"] = config_bb


### ======================= DBB H ========================
config_bbh = deepcopy(config)

## experiments configuration
# method
config_bbh.mthd = "bb"
# loss
config_bbh.loss = "h"
# time limit
config_bbh.timeout = 32

## training configuration
# size of neural network hidden layers
config_bbh.net = []
# number of epochs
config_bbh.batch = 32
# number of epochs
config_bbh.epoch = None
# optimizer neural network
config_bbh.optm = "adam"
# learning rate
config_bbh.lr = 5e-2
# smoothing parameter for Black-Box
config_bbh.smth = 10
# l1 regularization parameter
config_bbh.l1 = 0.0
# l2 regularization parameter
config_bbh.l2 = 0.0
# number of processor for optimization
config_bbh.proc = 32

configKS["bbh"] = config_bbh
