#!/usr/bin/env python
# coding: utf-8
"""
Tavelling salesman configuration
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
config.items = 48
# dimension
config.dim = 3
# capacity
config.cap = 30

## experiments configuration
# seed
config.seed = 135
# number of experiments
config.expnum = 10
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
# time limit
config_2s.timeout = 5 * config.expnum

configKS["2s"] = config_2s

### ========================= SPO =========================
config_spo = deepcopy(config)

## experiments configuration
# method
config_spo.mthd = "spo"
# time limit
config_spo.timeout = 30 * config.expnum

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
config_spo.lr = 5e-3
# l1 regularization parameter
config_spo.l1 = 0.0
# l2 regularization parameter
config_spo.l2 = 0.0
# number of processor for optimization
config_spo.proc = 8

configKS["spo"] = config_spo


### ========================= BB =========================
config_bb = deepcopy(config)

## experiments configuration
# method
config_bb.mthd = "bb"
# time limit
config_bb.timeout = 75 * config.expnum

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
config_bb.lr = 1e-3
# smoothing parameter for Black-Box
config_bb.smth = 20
# l1 regularization parameter
config_bb.l1 = 0.0
# l2 regularization parameter
config_bb.l2 = 0.0
# number of processor for optimization
config_bb.proc = 8

configKS["bb"] = config_bb
