#!/usr/bin/env python
# coding: utf-8
"""
Experiments
"""

import os
#os.chdir("./pkg") # set work dir
import argparse
import itertools
import sys

from config import configs
from pipeline import pipeline

# set parser
parser = argparse.ArgumentParser()
parser.add_argument("--prob",
                    type=str,
                    choices=["sp", "ks", "tsp"],
                    help="problem type")
parser.add_argument("--mthd",
                    type=str,
                    choices=["auto", "lr", "rf", "spo", "dbb"],
                    help="method")
parser.add_argument("--ksdim",
                    type=int,
                    default=2,
                    help="knapsack dimension")
parser.add_argument("--tspform",
                    type=str,
                    choices=["gg", "dfj", "mtz"],
                    help="TSP formulation")
parser.add_argument("--rel",
                    action="store_true",
                    help="train with relaxation model")
parser.add_argument("--l1",
                    action="store_true",
                    help="L1 regularization")
parser.add_argument("--l2",
                    action="store_true",
                    help="L2 regularization")
parser.add_argument("--expnum",
                    type=int,
                    default=10,
                    help="number of experiments")
setting = parser.parse_args()

# get config
config = configs[setting.prob][setting.mthd]
config.expnum = setting.expnum
if setting.prob == "ks":
    config.dim = setting.ksdim
if setting.prob == "tsp":
    config.form = setting.tspform
if setting.mthd in ["auto", "lr", "rf"]:
    config.mthd = "2s"
    config.pred = setting.mthd
config.rel = setting.rel
if setting.l1:
    config.l1 = 1e-3
if setting.l2:
    config.l2 = 1e-3


# config setting
confset = {"data":[100, 1000, 5000],
           "noise":[0.0, 0.5],
           "deg":[1, 2, 4, 6]}

for data, noise, deg in itertools.product(*tuple(confset.values())):
    # set config
    config.data = data
    config.noise = noise
    config.deg = deg
    if (setting.mthd != "2s") and (data == 5000):
        config.epoch = 4
    if (setting.mthd != "2s") and (data == 1000):
        config.epoch = 20
    if (setting.mthd != "2s") and (data == 100):
        config.epoch = 200
    print("===================================================================")
    print("===================================================================")
    print()
    print("Experiments configuration:")
    print(config)
    print()
    pipeline(config)
    print("===================================================================")
    print("===================================================================")
    print()
    print()

# python3 experiments.py --prob sp --mthd lr
# python3 experiments.py --prob sp --mthd rf
# python3 experiments.py --prob sp --mthd auto
# python3 experiments.py --prob sp --mthd spo
# python3 experiments.py --prob sp --mthd dbb
