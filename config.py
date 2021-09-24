#!/usr/bin/env python
# coding: utf-8
"""
Experiment configuration
"""

from types import SimpleNamespace

config = SimpleNamespace()

## experiments configuration
# seed
config.seed = 135
# method
config.mthd = "spo"
# number of experiments
config.expnum = 10
# relaxation
config.rel = False
# steps of evluation and log
config.elog = 0
# path to save result
config.path='./res'

## solver configuration
# modelling language
config.lan = 'gurobi'
# solver for Pyomo
#config.solver = "gurobi"

## data configuration
# training data size
config.data = 1000
# feature size
config.feat = 5
# features polynomial degree
config.deg = 1
# noise half-width
config.noise = 0.0

## optimization model configuration
# problem type
config.prob = 'sp'
# network grid for shortest path
config.grid = (20, 20)

## training configuration
# size of neural network hidden layers
config.net = []
# number of epochs
config.batch = 32
# number of epochs
config.epoch = 100
# optimizer neural network
config.optm = 'adam'
# learning rate
config.lr = 1e-2
# l1 regularization parameter
config.l1 = 0.0
# l2 regularization parameter
config.l2 = 0.0
# number of processor for optimization
config.proc = 1
