#!/usr/bin/env python
# coding: utf-8
"""
Plot
"""

from copy import deepcopy
from types import SimpleNamespace
import itertools

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.style.reload_library()
plt.style.use("science")
from matplotlib import ticker
import tol_colors as tc

from run import utils
from config import configs


def getConfig(prob):
    config = {}
    config["lr"]   = configs[prob]["lr"]
    config["rf"]   = configs[prob]["rf"]
    config["auto"] = configs[prob]["auto"]
    config["spo"]  = configs[prob]["spo"]
    config["dbb"]  = configs[prob]["dbb"]
    return config


def getDf(config, degs, mthd):
    dfs = pd.DataFrame()
    for deg in degs:
        config[mthd].deg = deg
        path = utils.getSavePath(config[mthd])
        df = pd.read_csv(path)
        dfs[deg] = df["True SPO"]
    return dfs


def comparisonPlot(config, data, noise):
    # polynomial degree
    degs = [1, 2, 4, 6]
    # set config
    config = deepcopy(config)
    for c in config.values():
        c.data = data
        c.noise = noise
    # color map
    cset =  tc.tol_cset('light')
    colors = [cset.mint, cset.pink, cset.pear, cset.orange, cset.light_blue]
    # automl flag
    automl = True
    # init box & label
    boxes = []
    label = []
    # get df
    df_lr = getDf(config, degs, "lr")
    df_rf = getDf(config, degs, "rf")
    try:
        df_auto  = getDf(config, degs, "auto")
    except:
        automl = False # no automl data
    df_spo = getDf(config, degs, "spo")
    df_dbb = getDf(config, degs, "dbb")
    # draw boxplot
    fig = plt.figure(figsize=(16,6))
    ########################################################################################################################
    c = colors[0]
    bp = plt.boxplot(df_lr, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                     whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                     flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                     patch_artist=True, positions=np.arange(df_spo.shape[1])-0.38, widths=0.16)
    boxes.append(bp["boxes"][0])
    label.append("2-stage LR")
    ########################################################################################################################
    c = colors[1]
    bp = plt.boxplot(df_rf, boxprops=dict(facecolor=c, color=c, linewidth=4),
                     medianprops=dict(color="w", alpha=0.9, linewidth=2),
                     whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                     flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                     patch_artist=True, positions=np.arange(df_spo.shape[1])-0.19, widths=0.16)
    boxes.append(bp["boxes"][0])
    label.append("2-stage RF")
    ########################################################################################################################
    if automl:
        c = colors[2]
        bp = plt.boxplot(df_auto, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                         whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                         flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                         patch_artist=True, positions=np.arange(df_spo.shape[1]), widths=0.16)
        boxes.append(bp["boxes"][0])
        label.append("2-stage Auto  ")
    ########################################################################################################################
    c = colors[3]
    bp = plt.boxplot(df_spo, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                     whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                     flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                     patch_artist=True, positions=np.arange(df_spo.shape[1])+0.19, widths=0.16)
    boxes.append(bp["boxes"][0])
    label.append("SPO+")
    ########################################################################################################################
    c = colors[4]
    bp = plt.boxplot(df_dbb, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                     whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                     flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                     patch_artist=True, positions=np.arange(df_spo.shape[1])+0.38, widths=0.16)
    boxes.append(bp["boxes"][0])
    label.append("DBB")
    ########################################################################################################################
    # vertical line
    plt.axvline(x=0.5, color="k", linestyle="--", linewidth=1.5)
    plt.axvline(x=1.5, color="k", linestyle="--", linewidth=1.5)
    plt.axvline(x=2.5, color="k", linestyle="--", linewidth=1.5)
    # grid
    plt.grid(color="grey", alpha=0.5, linewidth=0.5, which="major", axis="y")
    # labels and ticks
    plt.xlabel("Polynomial Degree", fontsize=36)
    plt.xticks(ticks=[0,1,2,3], labels=[1,2,4,6], fontsize=28)
    plt.ylabel("Normalized Regret", fontsize=36)
    plt.yticks(fontsize=24)
    plt.xlim(-0.5, 3.5)
    plt.ylim(-0.02, 0.38)
    plt.title("Training Set Size = {},\nNoise Half−width = {}".format(data, noise), fontsize=30)
    plt.legend(boxes, label, fontsize=22, loc=2, labelspacing=0.2, handlelength=1, ncol=2)
    # save
    dir = "./images/sp-n{}e{}.png".format(data,int(10*noise))
    fig.savefig(dir, dpi=300)
    print("Saved to " + dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--plot",
                        type=str,
                        choices=["comp"],
                        help="figure type")
    parser.add_argument("--prob",
                        type=str,
                        default="sp",
                        choices=["sp", "ks", "tsp"],
                        help="problem type")

    # get plot setting
    setting = parser.parse_args()

    if setting.plot == "comp":
        # get config
        config = getConfig(setting.prob)
        # varying setting
        confset = {"data":[100, 1000, 5000],
                   "noise":[0.0, 0.5]}
        for data, noise in itertools.product(*tuple(confset.values())):
            comparisonPlot(config, data, noise)

# python3 plot.py --plot comp --prob sp
