#!/usr/bin/env python
# coding: utf-8
"""
Tavelling salesman configuration
"""

from copy import deepcopy
from types import SimpleNamespace

# init configs
config = SimpleNamespace()
configTSP = {}


### ========================= general setting =========================
## optimization model configuration
# problem type
config.prob = "tsp"
# number of nodes
config.nodes = 20

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
config.feat = 10
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
config_lr.timeout = 10

configTSP["lr"] = config_lr

### ========================= rf =========================
config_rf = deepcopy(config)

## experiments configuration
# method
config_rf.mthd = "2s"
# predictor
config_rf.pred = "rf"
# time limit
config_rf.timeout = 20

configTSP["rf"] = config_rf

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
config_auto.timeout = 24

configTSP["auto"] = config_auto

### ========================= SPO =========================
config_spo = deepcopy(config)

## experiments configuration
# method
config_spo.mthd = "spo"
# time limit
config_spo.timeout = 10

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

configTSP["spo"] = config_spo


### ======================== DBB =========================
config_dbb = deepcopy(config)

## experiments configuration
# method
config_dbb.mthd = "dbb"
# loss
config_dbb.loss = "r"
# time limit
config_dbb.timeout = 15

## training configuration
# size of neural network hidden layers
config_dbb.net = []
# number of epochs
config_dbb.batch = 32
# number of epochs
config_dbb.epoch = None
# optimizer neural network
config_dbb.optm = "adam"
# learning rate
config_dbb.lr = 1e-1
# smoothing parameter for Black-Box
config_dbb.smth = 20
# l1 regularization parameter
config_dbb.l1 = 0.0
# l2 regularization parameter
config_dbb.l2 = 0.0
# number of processor for optimization
config_dbb.proc = 32

configTSP["dbb"] = config_dbb
