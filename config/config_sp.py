#!/usr/bin/env python
# coding: utf-8
"""
Shortest Path configuration
"""

from copy import deepcopy
from types import SimpleNamespace

# init configs
config = SimpleNamespace()
configSP = {}


### ========================= general setting =========================
## optimization model configuration
# problem type
config.prob = "sp"
# network grid for shortest path
config.grid = (5, 5)

## experiments configuration
# seed
config.seed = 135
# number of experiments
config.expnum = 1
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


### ========================= 2S =========================
config_2s = deepcopy(config)

## experiments configuration
# method
config_2s.mthd = "2s"
# predictor
config_2s.pred = None

configSP["2s"] = config_2s

### ========================= SPO =========================
config_spo = deepcopy(config)

## experiments configuration
# method
config_spo.mthd = "spo"

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
config_spo.lr = 1e-3
# l1 regularization parameter
config_spo.l1 = 0.0
# l2 regularization parameter
config_spo.l2 = 0.0
# number of processor for optimization
config_spo.proc = 32

configSP["spo"] = config_spo


### ========================= BB =========================
config_bb = deepcopy(config)

## experiments configuration
# method
config_bb.mthd = "bb"

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
config_bb.lr = 5e-3
# smoothing parameter for Black-Box
config_bb.smth = 20
# l1 regularization parameter
config_bb.l1 = 0.0
# l2 regularization parameter
config_bb.l2 = 0.0
# number of processor for optimization
config_bb.proc = 32

configSP["bb"] = config_bb
