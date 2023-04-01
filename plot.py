#!/usr/bin/env python
# coding: utf-8
"""
Plot
"""

import os
from copy import deepcopy
from types import SimpleNamespace
import itertools

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.style.reload_library()
plt.style.use("science")
from matplotlib import patheffects as path_effects
from matplotlib import ticker
from matplotlib.colors import colorConverter
import tol_colors as tc

from run import utils
from config import configs


def getConfig(prob):
    config = {}
    config["auto"] = configs[prob]["auto"]
    config["rf"]   = configs[prob]["rf"]
    config["pfyl"] = configs[prob]["pfyl"]
    config["spo"]  = configs[prob]["spo"]
    config["dbb"]  = configs[prob]["dbb"]
    config["dpo"]  = configs[prob]["dpo"]
    config["lr"]   = configs[prob]["lr"]
    return config


def getDf(config, degs, mthd, col="True SPO"):
    dfs = pd.DataFrame()
    for deg in degs:
        config[mthd].deg = deg
        path = utils.getSavePath(config[mthd])
        df = pd.read_csv(path)
        dfs[deg] = df[col]
    return dfs


def getElapsed(config):
    # polynomial degree
    degs = [1, 2, 4, 6]
    # init data
    df = pd.DataFrame(columns=["Method", "Elapsed_mean", "Elapsed_std"])
    # stat
    for mthd in config:
        elapses = np.empty((1,0))
        for noise in [0.0, 0.5]:
            config = deepcopy(config)
            # add noise
            for c in config.values():
                c.noise = noise
            for data in [100, 1000, 5000]:
                # add data size
                for c in config.values():
                    c.data = data
                # get df
                cur_df = getDf(config, degs, mthd, "Elapsed").to_numpy()
                # per iter
                cur_df /= getDf(config, degs, mthd, "Epochs").to_numpy() * np.ceil(data / config[mthd].batch)
                # get elapse
                elapses = np.concatenate((elapses, cur_df.reshape(1,-1)), axis=1)
        # stat
        elapsed_mean = elapses.mean()
        elapsed_std = elapses.std()
        row = {"Method":mthd, "Elapsed_mean":elapsed_mean, "Elapsed_std":elapsed_std}
        df = df.append(row, ignore_index=True)
    return df


def getRow(config, params, reg):
    regrets = {"spo":pd.DataFrame(), "pfyl":pd.DataFrame(),
               "dbb":pd.DataFrame(), "lr":pd.DataFrame()}
    mses = {"spo":pd.DataFrame(), "pfyl":pd.DataFrame(),
            "dbb":pd.DataFrame(), "lr":pd.DataFrame()}
    # go through different l1/l2 param
    for param in params:
        # add reg term
        if reg == "l1":
            for c in config.values():
                c.l1 = param
        if reg == "l2":
            for c in config.values():
                c.l2 = param
        # get data
        r, s = getRegData(config, reg)
        for m in regrets:
            regrets[m][param] = r[m]
            mses[m][param] = s[m]
    return regrets, mses


def getRegData(config, reg):
    regret, mse = {}, {}
    for mthd in config:
        path = utils.getSavePath(config[mthd])
        df = pd.read_csv(path)
        regret[mthd] = df["Unamb SPO"]
        mse[mthd] = df["MSE"]
    return regret, mse


def lighten(color, amount=0.9):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))
    return colorsys.hls_to_rgb(c[0], 1-amount*(1-c[1]), c[2])


def darken(color, amount=0.9):
    """
    Darkens the given color by multiplying (1-luminosity) by the given amount.
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))
    return colorsys.hls_to_rgb(c[0], amount*c[1], c[2])


def comparisonPlot(config, data, noise):
    # polynomial degree
    degs = [1, 2, 4, 6]
    # set config
    config = deepcopy(config)
    for c in config.values():
        c.data = data
        c.noise = noise
    # prob name
    prob = c.prob
    # color map
    cset =  tc.tol_cset('light')
    cmap = tc.tol_cmap("rainbow_discrete")(np.linspace(0, 1, 22))
    colors = [cset.mint, cset.pink, cset.pear, cmap[16], cmap[5], cmap[7], cmap[10]]
    for i in range(len(colors)):
        colors[i] = lighten(colors[i])
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
    df_pfyl = getDf(config, degs, "pfyl")
    df_dbb = getDf(config, degs, "dbb")
    try:
        df_dpo  = getDf(config, degs, "dpo")
    except:
        dpo = False # no dpo data
    # draw boxplot
    fig = plt.figure(figsize=(16,6))
    ########################################################################################################################
    c = colors[0]
    bp = plt.boxplot(df_lr, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                     whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                     flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                     patch_artist=True, positions=np.arange(df_spo.shape[1])-0.42, widths=0.11)
    boxes.append(bp["boxes"][0])
    label.append("2-stage LR")
    ########################################################################################################################
    c = colors[1]
    bp = plt.boxplot(df_rf, boxprops=dict(facecolor=c, color=c, linewidth=4),
                     medianprops=dict(color="w", alpha=0.9, linewidth=2),
                     whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                     flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                     patch_artist=True, positions=np.arange(df_spo.shape[1])-0.28, widths=0.11)
    boxes.append(bp["boxes"][0])
    label.append("2-stage RF")
    ########################################################################################################################
    if automl:
        c = colors[2]
        bp = plt.boxplot(df_auto, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                         whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                         flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                         patch_artist=True, positions=np.arange(df_spo.shape[1])-0.14, widths=0.11)
        boxes.append(bp["boxes"][0])
        label.append("2-stage Auto  ")
    ########################################################################################################################
    c = colors[3]
    bp = plt.boxplot(df_spo, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                     whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                     flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                     patch_artist=True, positions=np.arange(df_spo.shape[1])+0.00, widths=0.11)
    boxes.append(bp["boxes"][0])
    label.append("SPO+")
    ########################################################################################################################
    c = colors[4]
    bp = plt.boxplot(df_pfyl, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                     whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                     flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                     patch_artist=True, positions=np.arange(df_spo.shape[1])+0.14, widths=0.11)
    boxes.append(bp["boxes"][0])
    label.append("PFYL")
    ########################################################################################################################
    c = colors[5]
    bp = plt.boxplot(df_dbb, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                     whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                     flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                     patch_artist=True, positions=np.arange(df_spo.shape[1])+0.28, widths=0.11)
    boxes.append(bp["boxes"][0])
    label.append("DBB")
    ########################################################################################################################
    if dpo:
        c = colors[6]
        bp = plt.boxplot(df_dpo, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                         whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                         flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                         patch_artist=True, positions=np.arange(df_spo.shape[1])+0.42, widths=0.11)
        boxes.append(bp["boxes"][0])
        label.append("2-stage Auto  ")
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
    # title
    if prob == "sp":
        plt.title("Shortest Path\nTraining Set Size = {}, Noise Half−width = {}".format(data, noise), fontsize=30)
    if prob == "ks":
        plt.title("Knapsack\nTraining Set Size = {}, Noise Half−width = {}".format(data, noise), fontsize=30)
    if prob == "tsp":
        plt.title("TSP\nTraining Set Size = {}, Noise Half−width = {}".format(data, noise), fontsize=30)
    plt.legend(boxes, label, fontsize=22, loc=2, labelspacing=0.2, handlelength=1, ncol=3)
    # save
    dir = "./images/{}-n{}e{}.png".format(prob, data, int(10*noise))
    fig.savefig(dir, dpi=300)
    print("Saved to " + dir)
    plt.close()


def timePlotKS(config):
    # get mean & std
    df = getElapsed(config)
    # list
    means, stds = [], []
    for mthd in ["spo", "spo rel", "pfyl", "pfyl rel", "dbb", "dbb rel"]:#, "dpo", "dpo rel"]:
        means.append(df[df["Method"]==mthd]["Elapsed_mean"].values[0])
        stds.append(df[df["Method"]==mthd]["Elapsed_std"].values[0])
    # color map
    cmap = tc.tol_cmap("rainbow_discrete")(np.linspace(0, 1, 22))
    colors = [cmap[16], cmap[14], cmap[5], cmap[3], cmap[7], cmap[9], cmap[10], cmap[12]]
    for i in range(len(colors)):
        colors[i] = lighten(colors[i])
    # draw boxplot
    fig = plt.figure(figsize=(16,6))
    x = np.array(range(6))
    plt.bar(x, height=means, width=0.6, edgecolor="w",
                   linewidth=3, color=colors, label="Training Set Size = 1000")
    plt.errorbar(x, means, yerr=stds, capsize=5, capthick=2, linestyle="", marker="o",
                 markersize=3, color="k", elinewidth=2, alpha=0.7)
    # grid
    plt.grid(color="grey", alpha=0.5, linewidth=0.5, which="major", axis="y")
    # labels and ticks
    plt.xlim(-0.5, 5.5)
    plt.ylim(0.0, 0.5)
    plt.xticks(ticks=x, fontsize=22,
               labels=["SPO+\n", "SPO+ Rel\n", "PFYL\n", "PFYL Rel\n", "DBB\n", "DBB Rel\n"])
    plt.xlabel("Method", fontsize=36)
    plt.ylabel("Runtime per Iter (Sec)", fontsize=36)
    plt.yticks(fontsize=24)
    plt.title("2D Knapsack", fontsize=30)
    # save
    dir = "./images/rel-ks2-time.png"
    fig.savefig(dir, dpi=300)
    print("Saved to " + dir)
    plt.close()


def relPlotKS(config, data, noise):
    # polynomial degree
    degs = [1, 2, 4, 6]
    # set config
    config = deepcopy(config)
    for c in config.values():
        c.data = data
        c.noise = noise
    # get df
    df_spo      = getDf(config, degs, "spo")
    df_spo_rel  = getDf(config, degs, "spo rel")
    df_dbb      = getDf(config, degs, "dbb")
    df_dbb_rel  = getDf(config, degs, "dbb rel")
    #df_dpo      = getDf(config, degs, "dpo")
    #df_dpo_rel  = getDf(config, degs, "dpo rel")
    df_pfyl     = getDf(config, degs, "pfyl")
    df_pfyl_rel = getDf(config, degs, "pfyl rel")
    # color map
    cmap = tc.tol_cmap("rainbow_discrete")(np.linspace(0, 1, 22))
    colors = [cmap[16], cmap[14], cmap[5], cmap[3], cmap[7], cmap[9], cmap[10], cmap[12]]
    for i in range(len(colors)):
        colors[i] = lighten(colors[i])
    # draw boxplot
    fig = plt.figure(figsize=(16,6))
    #######################################################################################################################
    c = colors[0]
    bp1 = plt.boxplot(df_spo, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_spo.shape[1])-0.40, widths=0.12)
    c = colors[1]
    bp2 = plt.boxplot(df_spo_rel, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_spo.shape[1])-0.24, widths=0.12)
    #######################################################################################################################
    c = colors[2]
    bp3 = plt.boxplot(df_pfyl, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_dbb.shape[1])-0.08, widths=0.12)
    c = colors[3]
    bp4 = plt.boxplot(df_pfyl_rel, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_dbb.shape[1])+0.08, widths=0.12)
    #######################################################################################################################
    c = colors[4]
    bp5 = plt.boxplot(df_dbb, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_dbb.shape[1])+0.24, widths=0.12)
    c = colors[5]
    bp6 = plt.boxplot(df_dbb_rel, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_dbb.shape[1])+0.40, widths=0.12)
    # vertical line
    plt.axvline(x=0.5, color="k", linestyle="--", linewidth=1, alpha=0.75)
    plt.axvline(x=1.5, color="k", linestyle="--", linewidth=1, alpha=0.75)
    plt.axvline(x=2.5, color="k", linestyle="--", linewidth=1, alpha=0.75)
    # grid
    plt.grid(color="grey", alpha=0.5, linewidth=0.5, which="major", axis="y")
    # labels and ticks
    plt.xlabel("Polynomial Degree", fontsize=36)
    plt.xticks(ticks=range(len(degs)), labels=degs, fontsize=28)
    plt.ylabel("Normalized Regret", fontsize=36)
    plt.yticks(fontsize=24)
    plt.xlim(-0.5, 3.5)
    plt.ylim(-0.01, 0.34)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    plt.title("2D Knapsack\nTraining Set Size = {}, Noise Half−width = {}".format(data, noise), fontsize=30)
    plt.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0], bp4["boxes"][0], bp5["boxes"][0], bp6["boxes"][0]],
               ["SPO+", "SPO+ Rel", "PFYL", "PFYL Rel", "DBB", "DBB Rel"],
               fontsize=22, loc=2, labelspacing=0.2, handlelength=1, ncol=3)
    # save
    dir = "./images/rel-ks2-n{}e{}.png".format(data,int(10*noise))
    fig.savefig(dir, dpi=300)
    print("Saved to " + dir)
    plt.close()


def timePlotTSP(config):
    # get mean & std
    df = getElapsed(config)
    # list
    means, stds = [], []
    for mthd in ["spo", "spo rel(gg)", "spo rel(mtz)",
                 "pfyl", "pfyl rel(gg)", "pfyl rel(mtz)",
                 "dbb", "dbb rel(gg)", "dbb rel(mtz)"]:#, "dpo", "dpo rel"]:
        means.append(df[df["Method"]==mthd]["Elapsed_mean"].values[0])
        stds.append(df[df["Method"]==mthd]["Elapsed_std"].values[0])
    # color map
    cmap = tc.tol_cmap("rainbow_discrete")(np.linspace(0, 1, 22))
    colors = [cmap[16], cmap[15], cmap[14], cmap[5], cmap[4], cmap[3],
              cmap[7], cmap[8], cmap[9], cmap[10], cmap[11], cmap[12]]
    for i in range(len(colors)):
        colors[i] = lighten(colors[i])
    # draw boxplot
    fig = plt.figure(figsize=(16,6))
    x = np.array(range(9))
    bar = plt.bar(x, height=means, width=0.6, edgecolor="w",
                   linewidth=3, color=colors, label="Training Set Size = 1000")
    plt.errorbar(x, means, yerr=stds, capsize=5, capthick=2, linestyle="", marker="o",
                 markersize=3, color="k", elinewidth=2, alpha=0.7)
    # grid
    plt.grid(color="grey", alpha=0.5, linewidth=0.5, which="major", axis="y")
    # labels and ticks
    plt.xlim(-0.5, 8.5)
    plt.ylim(0.0, 1.6)
    plt.xticks(ticks=x, fontsize=22,
               labels=["SPO+\n(DFJ)", "SPO+ Rel\n(GG)", "SPO+ Rel\n(MTZ)",
                       "PFYL\n(DFJ)", "PFYL Rel\n(GG)", "PFYL Rel\n(MTZ)",
                       "DBB\n(DFJ)", "DBB Rel\n(GG)", "DBB Rel\n(MTZ)"])
    plt.xlabel("Method", fontsize=36)
    plt.ylabel("Runtime per Iter (Sec)", fontsize=36)
    plt.yticks(fontsize=24)
    plt.title("TSP", fontsize=30)
    # save
    dir = "./images/rel-tsp-time.png"
    fig.savefig(dir, dpi=300)
    print("Saved to " + dir)
    plt.close()


def relPlotTSP(config, data, noise):
    # polynomial degree
    degs = [1, 2, 4, 6]
    # set config
    config = deepcopy(config)
    for c in config.values():
        c.data = data
        c.noise = noise
    # get df
    df_spo_dfj  = getDf(config, degs, "spo")
    df_spo_gg  = getDf(config, degs, "spo rel(gg)")
    df_spo_mtz = getDf(config, degs, "spo rel(mtz)")
    df_pfyl_dfj  = getDf(config, degs, "pfyl")
    df_pfyl_gg  = getDf(config, degs, "pfyl rel(gg)")
    df_pfyl_mtz = getDf(config, degs, "pfyl rel(mtz)")
    df_dbb_dfj  = getDf(config, degs, "dbb")
    df_dbb_gg  = getDf(config, degs, "dbb rel(gg)")
    df_dbb_mtz = getDf(config, degs, "dbb rel(mtz)")
    # color map
    cmap = tc.tol_cmap("rainbow_discrete")(np.linspace(0, 1, 22))
    colors = [cmap[16], cmap[15], cmap[14], cmap[5], cmap[4], cmap[3],
              cmap[7], cmap[8], cmap[9], cmap[10], cmap[11], cmap[12]]
    for i in range(len(colors)):
        colors[i] = lighten(colors[i])
    # draw boxplot
    fig = plt.figure(figsize=(16,6))
    #######################################################################################################################
    c = colors[0]
    bp1 = plt.boxplot(df_spo_dfj, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_spo_dfj.shape[1])-0.44, widths=0.08)
    c = colors[1]
    bp2 = plt.boxplot(df_spo_gg, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_spo_dfj.shape[1])-0.33, widths=0.08)
    c = colors[2]
    bp3 = plt.boxplot(df_spo_mtz, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_spo_dfj.shape[1])-0.22, widths=0.08)
    #######################################################################################################################
    c = colors[3]
    bp4 = plt.boxplot(df_pfyl_dfj, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_dbb_dfj.shape[1])-0.11, widths=0.08)
    c = colors[4]
    bp5 = plt.boxplot(df_pfyl_gg, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_dbb_dfj.shape[1]), widths=0.08)
    c = colors[5]
    bp6 = plt.boxplot(df_pfyl_mtz, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_dbb_dfj.shape[1])+0.11, widths=0.08)
    #######################################################################################################################
    c = colors[6]
    bp7 = plt.boxplot(df_dbb_dfj, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_dbb_dfj.shape[1])+0.22, widths=0.08)
    c = colors[7]
    bp8= plt.boxplot(df_dbb_gg, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_dbb_dfj.shape[1])+0.33, widths=0.08)
    c = colors[8]
    bp9= plt.boxplot(df_dbb_mtz, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_dbb_dfj.shape[1])+0.44, widths=0.08)
    # vertical line
    plt.axvline(x=0.5, color="k", linestyle="--", linewidth=1, alpha=0.75)
    plt.axvline(x=1.5, color="k", linestyle="--", linewidth=1, alpha=0.75)
    plt.axvline(x=2.5, color="k", linestyle="--", linewidth=1, alpha=0.75)
    # grid
    plt.grid(color="grey", alpha=0.5, linewidth=0.5, which="major", axis="y")
    # labels and ticks
    plt.xlabel("Polynomial Degree", fontsize=36)
    plt.xticks(ticks=[0,1,2,3], labels=[1,2,4,6], fontsize=28)
    plt.ylabel("Normalized Regret", fontsize=36)
    plt.yticks(fontsize=24)
    plt.xlim(-0.5, 3.5)
    plt.ylim(-0.01, 0.44)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    plt.title("TSP\nTraining Set Size = {}, Noise Half−width = {}".format(data, noise), fontsize=30)
    plt.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0], bp4["boxes"][0],
                bp5["boxes"][0], bp6["boxes"][0], bp7["boxes"][0], bp8["boxes"][0], bp9["boxes"][0]],
               ["SPO+ (DFJ)", "SPO+ Rel(GG)", "SPO+ Rel(MTZ)", "PFYL (DFJ)", "PFYL Rel(GG)",
                "PFYL Rel(MTZ)", "DBB (DFJ)", "DBB Rel(GG)", "DBB Rel(MTZ)"],
               fontsize=22, loc=2, labelspacing=0.2, handlelength=1, ncol=3)
    # save
    dir = "./images/rel-tsp-n{}e{}.png".format(data,int(10*noise))
    fig.savefig(dir, dpi=300)
    print("Saved to " + dir)
    plt.close()


def regPlot(config, data, deg, noise, reg):
    config = deepcopy(config)
    for c in config.values():
        c.data = data
        c.noise = noise
        c.deg = deg
        prob = c.prob
    # get prob name
    if prob == "sp":
        prob_name = "Shortest Path"
    if prob == "ks":
        prob_name = "2D Knapsack"
    if prob == "tsp":
        prob_name = "TSP"
    # l1/l2 params
    params = [0.0, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    # get rows
    regrets, mses = getRow(config, params, reg)
    # color map
    cset =  tc.tol_cset('light')
    cmap = tc.tol_cmap("rainbow_discrete")(np.linspace(0, 1, 22))
    colors = [cset.mint, cset.pink, cmap[16], cmap[5], cmap[7], cmap[10]]
    for i in range(len(colors)):
        colors[i] = lighten(colors[i])
    # x tick
    x = np.array([i for i in range(len(params))])
    # figure
    fig = plt.figure(figsize=(32, 8))
    # regret
    ax1 = plt.subplot(121)
    # lr line
    c = colors[0]
    line = plt.plot(range(-1, 7), [regrets["lr"][0.0].mean()]*8, linewidth=4, color=c, linestyle="--")
    # line
    c = colors[2]
    plt.plot(x-0.24, regrets["spo"].mean(), linewidth=3, color=c)
    c = colors[3]
    plt.plot(x, regrets["pfyl"].mean(), linewidth=3, color=c)
    c = colors[4]
    plt.plot(x+0.24, regrets["dbb"].mean(), linewidth=3, color=c)
    # box plot
    #===========================================================================================================================
    c = colors[2]
    bp1 = plt.boxplot(regrets["spo"],
                      boxprops=dict(facecolor=c, color=c, linewidth=6),
                      medianprops=dict(color="w", linewidth=4),
                      whiskerprops=dict(color=c, linewidth=4),
                      capprops=dict(color=c, linewidth=4),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=8, markeredgewidth=3),
                      patch_artist=True, positions=np.arange(regrets["spo"].shape[1])-0.24, widths=0.06)
    #===========================================================================================================================
    c = colors[3]
    bp2 = plt.boxplot(regrets["pfyl"],
                      boxprops=dict(facecolor=c, color=c, linewidth=6),
                      medianprops=dict(color="w", linewidth=4),
                      whiskerprops=dict(color=c, linewidth=4),
                      capprops=dict(color=c, linewidth=4),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=8, markeredgewidth=3),
                      patch_artist=True, positions=np.arange(regrets["pfyl"].shape[1]), widths=0.06)
    #===========================================================================================================================
    c = colors[4]
    bp3 = plt.boxplot(regrets["dbb"],
                      boxprops=dict(facecolor=c, color=c, linewidth=6),
                      medianprops=dict(color="w", linewidth=4),
                      whiskerprops=dict(color=c, linewidth=4),
                      capprops=dict(color=c, linewidth=4),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=8, markeredgewidth=3),
                      patch_artist=True, positions=np.arange(regrets["dbb"].shape[1])+0.24, widths=0.06)
    #===========================================================================================================================
    # vertical line
    plt.axvline(x=0.5, color="k", linestyle="--", linewidth=1, alpha=0.75)
    plt.axvline(x=1.5, color="k", linestyle="--", linewidth=1, alpha=0.75)
    plt.axvline(x=2.5, color="k", linestyle="--", linewidth=1, alpha=0.75)
    plt.axvline(x=3.5, color="k", linestyle="--", linewidth=1, alpha=0.75)
    plt.axvline(x=4.5, color="k", linestyle="--", linewidth=1, alpha=0.75)
    # labels and ticks
    plt.xlim(-0.5, 5.5)
    plt.ylim(0, 0.32)
    plt.xticks(x, labels=params, fontsize=44)
    plt.yticks(fontsize=40)
    plt.ylabel("Normalized Regret", fontsize=60)
    plt.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0], line[0]],
               ["SPO+", "PFYL", "DBB", "2-Stage LR"],
               fontsize=40, ncol=2, loc=1)
    plt.title("", fontsize=48)
    # mse
    ax1 = plt.subplot(122)
    # lr line
    c = colors[0]
    line = plt.plot(range(-1, 7), [mses["lr"][0.0].mean()]*8, linewidth=4, color=c, linestyle="--")
    # line
    c = colors[2]
    plt.plot(x-0.24, mses["spo"].mean(), linewidth=3, color=c)
    c = colors[3]
    plt.plot(x, mses["pfyl"].mean(), linewidth=3, color=c)
    c = colors[4]
    plt.plot(x+0.24, mses["dbb"].mean(), linewidth=3, color=c)
    # boxplot
    #===========================================================================================================================
    c = colors[2]
    bp1 = plt.boxplot(mses["spo"],
                      boxprops=dict(facecolor=c, color=c, linewidth=6),
                      medianprops=dict(color="w", linewidth=4),
                      whiskerprops=dict(color=c, linewidth=4),
                      capprops=dict(color=c, linewidth=4),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=8, markeredgewidth=3),
                      patch_artist=True, positions=np.arange(mses["spo"].shape[1])-0.24, widths=0.06)
    #===========================================================================================================================
    c = colors[3]
    bp2 = plt.boxplot(mses["pfyl"],
                      boxprops=dict(facecolor=c, color=c, linewidth=6),
                      medianprops=dict(color="w", linewidth=4),
                      whiskerprops=dict(color=c, linewidth=4),
                      capprops=dict(color=c, linewidth=4),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=8, markeredgewidth=3),
                      patch_artist=True, positions=np.arange(mses["pfyl"].shape[1]), widths=0.06)
    #===========================================================================================================================
    c = colors[4]
    bp3 = plt.boxplot(mses["dbb"],
                      boxprops=dict(facecolor=c, color=c, linewidth=6),
                      medianprops=dict(color="w", linewidth=4),
                      whiskerprops=dict(color=c, linewidth=4),
                      capprops=dict(color=c, linewidth=4),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=8, markeredgewidth=3),
                      patch_artist=True, positions=np.arange(mses["dbb"].shape[1])+0.24, widths=0.06)
    #===========================================================================================================================
    # vertical line
    plt.axvline(x=0.5, color="k", linestyle="--", linewidth=1, alpha=0.75)
    plt.axvline(x=1.5, color="k", linestyle="--", linewidth=1, alpha=0.75)
    plt.axvline(x=2.5, color="k", linestyle="--", linewidth=1, alpha=0.75)
    plt.axvline(x=3.5, color="k", linestyle="--", linewidth=1, alpha=0.75)
    plt.axvline(x=4.5, color="k", linestyle="--", linewidth=1, alpha=0.75)
    # labels and ticks
    plt.xlim(-0.5, 5.5)
    if prob == "sp":
        plt.ylim(-1, 23)
    if prob == "ks":
        plt.ylim(-2, 59)
    plt.xticks(x, labels=params, fontsize=44)
    plt.yticks(fontsize=40)
    plt.ylabel("MSE", fontsize=54)
    plt.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0], line[0]],
               ["SPO+", "PFYL", "DBB", "2-Stage LR"],
               fontsize=40, ncol=2, loc=1)
    # title
    plt.title("", fontsize=48)
    if reg == "l1":
        # xlabel
        fig.text(0.5, -0.02, "L1 Parameter", ha="center", va="center", fontsize=60)
        # title
        plt.suptitle("Test Loss on {} with L1 Regularization \
                     \nTraining Set Size = {}, Polynomial Degree = {}, Noise Half−width = {}".
                     format(prob_name, data, deg, noise),
                     y=1.1, fontsize=60)
    if reg == "l2":
        # xlabel
        fig.text(0.5, -0.02, "L2 Parameter", ha="center", va="center", fontsize=60)
        # title
        plt.suptitle("Test Loss on {} with L2 Regularization \
                     \nTraining Set Size = {}, Polynomial Degree = {}, Noise Half−width = {}".
                     format(prob_name, data, deg, noise),
                     y=1.1, fontsize=60)
    # xlabel
    if reg == "l1":
        fig.text(0.5, -0.02, "L1 Parameter", ha="center", va="center", fontsize=60)
    if reg == "l2":
        fig.text(0.5, -0.02, "L2 Parameter", ha="center", va="center", fontsize=60)
    # save
    dir = "./images/{}-{}-n{}d{}e{}.png".format(reg, prob, data, deg, int(10*noise))
    fig.savefig(dir, dpi=300)
    print("Saved to " + dir)
    plt.close()


def tradeoffPlot(config, data, noise, deg=4):
    # set config
    config = deepcopy(config)
    for c in config.values():
        c.data = data
        c.noise = noise
        c.deg = deg
    # prob name
    prob = c.prob
    # color map
    cset =  tc.tol_cset('light')
    cmap = tc.tol_cmap("rainbow_discrete")(np.linspace(0, 1, 22))
    colors = {"lr":lighten(cset.mint),
              "rf":lighten(cset.pink),
              "auto":lighten(cset.pear),
              "spo":lighten(cmap[16]),
              "spo rel":lighten(cmap[14]),
              "spo rel(gg)":lighten(cmap[15]),
              "spo rel(mtz)":lighten(cmap[14]),
              "pfyl":lighten(cmap[5]),
              "pfyl rel":lighten(cmap[3]),
              "pfyl rel(gg)":lighten(cmap[4]),
              "pfyl rel(mtz)":lighten(cmap[3]),
              "dbb":lighten(cmap[7]),
              "dbb rel":lighten(cmap[9]),
              "dbb rel(gg)":lighten(cmap[8]),
              "dbb rel(mtz)":lighten(cmap[9])}
    names =  {"lr":"2-stage LR",
              "rf":"2-stage RF",
              "auto":"2-stage Auto",
              "spo":"SPO+",
              "spo rel":"SPO+ Rel",
              "spo rel(gg)":"SPO+ Rel(GG)",
              "spo rel(mtz)":"SPO+ Rel(MTZ)",
              "pfyl":"PFYL",
              "pfyl rel":"PFYL Rel",
              "pfyl rel(gg)":"PFYL Rel(GG)",
              "pfyl rel(mtz)":"PFYL Rel(MTZ)",
              "dbb":"DBB",
              "dbb rel":"DBB Rel",
              "dbb rel(gg)":"DBB Rel(GG)",
              "dbb rel(mtz)":"DBB Rel(GG)",
              }
    w = colorConverter.to_rgba("w", alpha=0.6) # white
    k = colorConverter.to_rgba("k", alpha=0.5) # black
    # get df
    dfs = {}
    for mthd in config:
        path = utils.getSavePath(config[mthd])
        dfs[mthd] = pd.read_csv(path)
    # draw boxplot
    fig, ax = plt.subplots(figsize=(12,12))
    # init xmax & ymax & ymin
    xmax, ymax = 0, 0
    for mthd in dfs:
        df, c = dfs[mthd], colorConverter.to_rgba(colors[mthd], alpha=1.0)
        x, y = df["MSE"].mean(), df["True SPO"].mean()
        xmax, ymax = max(x, xmax), max(y, ymax)
        size = max(int(df["Elapsed"].mean() * 50), 50)
        ax.scatter(x, y, s=size, color=c, marker="o")
        # annotate
        if (names[mthd] == "SPO+ Rel") or (names[mthd] == "SPO+ Rel(MTZ)") or \
           (names[mthd] == "PFYL") or (names[mthd] == "PFYL Rel(GG)") or \
           (names[mthd] == "2-stage Auto"):
            txt = ax.annotate(names[mthd]+":{:.2f} Sec".format(df["Elapsed"].mean()), (x,y),
                              fontsize=24, color=darken(colors[mthd], 0.8), weight="black")
        else:
            txt = ax.annotate(names[mthd]+":{:.2f} Sec".format(df["Elapsed"].mean()), (x,y),
                              fontsize=24, color=darken(colors[mthd]), weight="black")
        txt.set_path_effects([path_effects.withStroke(linewidth=0.25, foreground=k),
                              path_effects.Normal()])
    plt.xlabel("Mean Squared Error", fontsize=36)
    plt.xticks(fontsize=24)
    plt.ylabel("Normalized Regret", fontsize=36)
    plt.yticks(fontsize=24)
    plt.xlim(-1.0, xmax*1.1)
    plt.ylim(0.05, ymax*1.1)
    plt.title("Training Set Size = {}, Polynomial degree = {}, Noise Half−width = {}" \
              .format(data, deg, noise), fontsize=24)
    # save
    dir = "./images/td-{}-n{}e{}.png".format(prob, data, int(10*noise))
    fig.savefig(dir, dpi=300)
    print("Saved to " + dir)
    plt.close()


def wcLearningCurve(regret_logs):
    # colors
    cmap = tc.tol_cmap("rainbow_discrete")(np.linspace(0, 1, 22))
    colors = {"2S": lighten(cmap[19]),
              "SPO+": lighten(cmap[16]),
              "DBB": lighten(cmap[7]),
              "DPO": lighten(cmap[10]),
              "PFYL": lighten(cmap[5])
              }
    # linestyles
    linestyles = {"2S": "-",
                  "SPO+": "--",
                  "DBB": "-.",
                  "DPO": ":",
                  "PFYL": (0,(3,1,1,1,1,1))
                  }
    # drow learning curve on test set
    fig = plt.figure(figsize=(10,6))
    for mthd in regret_logs:
        plt.plot(regret_logs[mthd], color=colors[mthd], lw=3, ls=linestyles[mthd], label=mthd)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(-1, 51)
    plt.ylim(-0.05, 0.85)
    plt.xlabel("Epochs", fontsize=36)
    plt.ylabel("Normalized Regret", fontsize=36)
    plt.title("Learning Curve on Test Set", fontsize=36)
    plt.legend(fontsize=22)
    # save
    dir = "./images/wc_lc.png"
    fig.savefig(dir, dpi=300)
    print("Saved to " + dir)
    plt.close()


def wcRegret(dfs):
    # colors
    cmap = tc.tol_cmap("rainbow_discrete")(np.linspace(0, 1, 22))
    colors = {"2S": lighten(cmap[19]),
              "SPO+": lighten(cmap[16]),
              "DBB": lighten(cmap[7]),
              "DPO": lighten(cmap[10]),
              "PFYL": lighten(cmap[5])
              }
    # draw boxplot of regret per instance
    fig = plt.figure(figsize=(10,6))
    boxplot_data, boxcolors = [], []
    for mthd in dfs:
        boxplot_data.append(dfs[mthd]["Regret"])
        boxcolors.append(colors[mthd])
    bp = plt.boxplot(boxplot_data, medianprops=dict(color="dimgrey", linewidth=2), patch_artist=True, widths=0.75)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(boxcolors[i])
        patch.set_color(boxcolors[i])
        patch.set_linewidth(4)
    for i, patch in enumerate(bp["whiskers"]):
        patch.set_color(boxcolors[i//2])
        patch.set_linewidth(2)
    for i, patch in enumerate(bp["caps"]):
        patch.set_color(boxcolors[i//2])
        patch.set_linewidth(3)
    for i, patch in enumerate(bp["fliers"]):
        patch.set_marker("o")
        patch.set_markeredgecolor(boxcolors[i])
        patch.set_markersize(6)
        patch.set_markeredgewidth(2)
    for i, patch in enumerate(bp["medians"]):
        patch.set_color("w")
        patch.set_linewidth(2)
    # grid
    plt.grid(color="grey", alpha=0.5, linewidth=0.5, which="major", axis="y")
    # labels and ticks
    plt.xticks(ticks=range(1, len(dfs)+1), fontsize=24, labels=dfs.keys())
    plt.xlabel("Methods", fontsize=36)
    plt.ylabel("Regret", fontsize=36)
    plt.yticks(fontsize=24)
    plt.xlim(0.5, 5.5)
    plt.ylim(-0.2, 40)
    plt.title("Regret for each Instance on Test Set", fontsize=36)
    # save
    dir = "./images/wc_reg.png"
    fig.savefig(dir, dpi=300)
    print("Saved to " + dir)
    plt.close()


def wcRelRegret(dfs):
    # colors
    cmap = tc.tol_cmap("rainbow_discrete")(np.linspace(0, 1, 22))
    colors = {"2S": lighten(cmap[19]),
              "SPO+": lighten(cmap[16]),
              "DBB": lighten(cmap[7]),
              "DPO": lighten(cmap[10]),
              "PFYL": lighten(cmap[5])
              }
    # draw boxplot of regret per instance
    fig = plt.figure(figsize=(10,6))
    boxplot_data, boxcolors = [], []
    for mthd in dfs:
        boxplot_data.append(dfs[mthd]["Relative Regret"])
        boxcolors.append(colors[mthd])
    bp = plt.boxplot(boxplot_data, medianprops=dict(color="dimgrey", linewidth=2), patch_artist=True, widths=0.75)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(boxcolors[i])
        patch.set_color(boxcolors[i])
        patch.set_linewidth(4)
    for i, patch in enumerate(bp["whiskers"]):
        patch.set_color(boxcolors[i//2])
        patch.set_linewidth(2)
    for i, patch in enumerate(bp["caps"]):
        patch.set_color(boxcolors[i//2])
        patch.set_linewidth(3)
    for i, patch in enumerate(bp["fliers"]):
        patch.set_marker("o")
        patch.set_markeredgecolor(boxcolors[i])
        patch.set_markersize(6)
        patch.set_markeredgewidth(2)
    for i, patch in enumerate(bp["medians"]):
        patch.set_color("w")
        patch.set_linewidth(2)
    # grid
    plt.grid(color="grey", alpha=0.5, linewidth=0.5, which="major", axis="y")
    # labels and ticks
    plt.xticks(ticks=range(1, len(dfs)+1), fontsize=24, labels=dfs.keys())
    plt.xlabel("Methods", fontsize=36)
    plt.ylabel("Relative Regret", fontsize=36)
    plt.yticks(fontsize=24)
    plt.xlim(0.5, 5.5)
    plt.ylim(-0.05, 1.8)
    plt.title("Relative Regret for each Instance on Test Set", fontsize=36)
    # save
    dir = "./images/wc_relreg.png"
    fig.savefig(dir, dpi=300)
    print("Saved to " + dir)
    plt.close()


def wcAcc(dfs):
    # colors
    cmap = tc.tol_cmap("rainbow_discrete")(np.linspace(0, 1, 22))
    colors = {"2S": lighten(cmap[19]),
              "SPO+": lighten(cmap[16]),
              "DBB": lighten(cmap[7]),
              "DPO": lighten(cmap[10]),
              "PFYL": lighten(cmap[5])
              }
    # draw boxplot of regret per instance
    fig = plt.figure(figsize=(10,6))
    boxplot_data, boxcolors = [], []
    for mthd in dfs:
        boxplot_data.append(dfs[mthd]["Accuracy"])
        boxcolors.append(colors[mthd])
    bp = plt.boxplot(boxplot_data, medianprops=dict(color="dimgrey", linewidth=2), patch_artist=True, widths=0.75)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(boxcolors[i])
        patch.set_color(boxcolors[i])
        patch.set_linewidth(4)
    for i, patch in enumerate(bp["whiskers"]):
        patch.set_color(boxcolors[i//2])
        patch.set_linewidth(2)
    for i, patch in enumerate(bp["caps"]):
        patch.set_color(boxcolors[i//2])
        patch.set_linewidth(3)
    for i, patch in enumerate(bp["fliers"]):
        patch.set_marker("o")
        patch.set_markeredgecolor(boxcolors[i])
        patch.set_markersize(6)
        patch.set_markeredgewidth(2)
    for i, patch in enumerate(bp["medians"]):
        patch.set_color("w")
        patch.set_linewidth(2)
    # grid
    plt.grid(color="grey", alpha=0.5, linewidth=0.5, which="major", axis="y")
    # labels and ticks
    plt.xticks(ticks=range(1, len(dfs)+1), fontsize=24, labels=dfs.keys())
    plt.xlabel("Methods", fontsize=36)
    plt.ylabel("Path Accuracy", fontsize=36)
    plt.yticks(fontsize=24)
    plt.xlim(0.5, 5.5)
    plt.ylim(0.6, 1.02)
    plt.title("Path Accuracy for each Instance on Test Set", fontsize=36)
    # save
    dir = "./images/wc_acc.png"
    fig.savefig(dir, dpi=300)
    print("Saved to " + dir)
    plt.close()


def wcOr(dfs):
    # colors
    cmap = tc.tol_cmap("rainbow_discrete")(np.linspace(0, 1, 22))
    colors = {"2S": lighten(cmap[19]),
              "SPO+": lighten(cmap[16]),
              "DBB": lighten(cmap[7]),
              "DPO": lighten(cmap[10]),
              "PFYL": lighten(cmap[5])
              }
    # draw boxplot of regret per instance
    fig = plt.figure(figsize=(10,6))
    barplot_data, barcolors = [], []
    for mthd in dfs:
        barplot_data.append(dfs[mthd]["Optimal"].mean())
        barcolors.append(colors[mthd])
    bp = plt.bar(range(len(dfs)), barplot_data, color=barcolors)
    # grid
    plt.grid(color="grey", alpha=0.5, linewidth=0.5, which="major", axis="y")
    # labels and ticks
    plt.xticks(ticks=range(len(dfs)), fontsize=24, labels=dfs.keys())
    plt.xlabel("Methods", fontsize=36)
    plt.ylabel("Ratio of the Optimality", fontsize=36)
    plt.yticks(fontsize=24)
    plt.title("Ratio of the Optimality on Test Set", fontsize=36)
    # save
    dir = "./images/wc_or.png"
    fig.savefig(dir, dpi=300)
    print("Saved to " + dir)
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--plot",
                        type=str,
                        choices=["cmp", "rel", "reg", "trd", "wc"],
                        help="figure type")
    parser.add_argument("--prob",
                        type=str,
                        default="sp",
                        choices=["sp", "ks", "tsp"],
                        help="problem type")

    # get plot setting
    setting = parser.parse_args()

    ############################################################################
    # performance comparison
    if setting.plot == "cmp":
        # get config
        config = getConfig(setting.prob)
        # varying setting
        confset = {"data":[100, 1000, 5000],
                   "noise":[0.0, 0.5]}
        # plot
        for data, noise in itertools.product(*tuple(confset.values())):
            comparisonPlot(config, data, noise)

    ############################################################################
    # relaxation
    if setting.plot == "rel":
        # get config
        config = getConfig(setting.prob)
        if setting.prob == "ks":
            # add relaxation
            config["spo rel"] = deepcopy(config["spo"])
            config["spo rel"].rel = True
            config["pfyl rel"] = deepcopy(config["pfyl"])
            config["pfyl rel"].rel = True
            config["dbb rel"] = deepcopy(config["dbb"])
            config["dbb rel"].rel = True
            # delete 2s
            del config["lr"]
            del config["rf"]
            del config["auto"]
            del config["dpo"]
            # runtime per epoch
            timePlotKS(config)
            # varying setting
            confset = {"data":[100, 1000, 5000],
                       "noise":[0.0, 0.5]}
            # plot
            for data, noise in itertools.product(*tuple(confset.values())):
                relPlotKS(config, data, noise)
        if setting.prob == "tsp":
            # add relaxation
            config["spo rel(mtz)"] = deepcopy(config["spo"])
            config["spo rel(mtz)"].form = "mtz"
            config["spo rel(mtz)"].rel = True
            config["pfyl rel(mtz)"] = deepcopy(config["pfyl"])
            config["pfyl rel(mtz)"].form = "mtz"
            config["pfyl rel(mtz)"].rel = True
            config["dbb rel(mtz)"] = deepcopy(config["dbb"])
            config["dbb rel(mtz)"].form = "mtz"
            config["dbb rel(mtz)"].rel = True
            config["spo rel(gg)"] = deepcopy(config["spo"])
            config["spo rel(gg)"].form = "gg"
            config["spo rel(gg)"].rel = True
            config["pfyl rel(gg)"] = deepcopy(config["pfyl"])
            config["pfyl rel(gg)"].form = "gg"
            config["pfyl rel(gg)"].rel = True
            config["dbb rel(gg)"] = deepcopy(config["dbb"])
            config["dbb rel(gg)"].form = "gg"
            config["dbb rel(gg)"].rel = True
            config["spo"].form = "dfj"
            config["pfyl"].form = "dfj"
            config["dbb"].form = "dfj"
            # delete 2s
            del config["lr"]
            del config["rf"]
            del config["auto"]
            # delete dpo
            del config["dpo"]
            # runtime per epoch
            timePlotTSP(config)
            # varying setting
            confset = {"data":[100, 1000],
                       "noise":[0.0, 0.5]}
            # plot
            for data, noise in itertools.product(*tuple(confset.values())):
                relPlotTSP(config, data, noise)

    ############################################################################
    # regularization
    if setting.plot == "reg":
        # get config
        config = getConfig(setting.prob)
        # delete 2s
        del config["rf"]
        del config["auto"]
        # delete dpo
        del config["dpo"]
        # varying setting
        confset = {"data":[100, 1000],
                   "deg": [2, 4, 6],
                   "noise":[0.5]}
        # plot
        for data, deg, noise in itertools.product(*tuple(confset.values())):
            regPlot(config, data, deg, noise, "l1")
            regPlot(config, data, deg, noise, "l2")

    ############################################################################
    # regularization
    if setting.plot == "trd":
        # get config
        config = getConfig(setting.prob)
        # delete dpo
        del config["dpo"]
        # add relaxation
        if setting.prob == "ks":
            config["spo rel"] = deepcopy(config["spo"])
            config["spo rel"].rel = True
            config["pfyl rel"] = deepcopy(config["pfyl"])
            config["pfyl rel"].rel = True
            config["dbb rel"] = deepcopy(config["dbb"])
            config["dbb rel"].rel = True
        if setting.prob == "tsp":
            # add relaxation
            config["spo rel(mtz)"] = deepcopy(config["spo"])
            config["spo rel(mtz)"].form = "mtz"
            config["spo rel(mtz)"].rel = True
            config["pfyl rel(mtz)"] = deepcopy(config["pfyl"])
            config["pfyl rel(mtz)"].form = "mtz"
            config["pfyl rel(mtz)"].rel = True
            config["dbb rel(mtz)"] = deepcopy(config["dbb"])
            config["dbb rel(mtz)"].form = "mtz"
            config["dbb rel(mtz)"].rel = True
            config["spo rel(gg)"] = deepcopy(config["spo"])
            config["spo rel(gg)"].form = "gg"
            config["spo rel(gg)"].rel = True
            config["pfyl rel(gg)"] = deepcopy(config["pfyl"])
            config["pfyl rel(gg)"].form = "gg"
            config["pfyl rel(gg)"].rel = True
            config["dbb rel(gg)"] = deepcopy(config["dbb"])
            config["dbb rel(gg)"].form = "gg"
            config["dbb rel(gg)"].rel = True
            config["spo"].form = "dfj"
            config["pfyl"].form = "dfj"
            config["dbb"].form = "dfj"
        # varying setting
        confset = {"data":[100, 1000],
                   "noise":[0.5]}
        # plot
        for data, noise in itertools.product(*tuple(confset.values())):
            tradeoffPlot(config, data, noise)

    ############################################################################
    # Warcraft
    if setting.plot == "wc":
        # load res
        regret_logs, dfs = {}, {}
        # 2s
        if os.path.isfile("./res/wc_2s.csv"):
            dfs["2S"] = pd.read_csv("./res/wc_2s.csv")
            regret_logs["2S"] = pd.read_csv("./res/log_2s.csv", header=None)
        # spo+
        if os.path.isfile("./res/wc_spo.csv"):
            dfs["SPO+"] = pd.read_csv("./res/wc_spo.csv")
            regret_logs["SPO+"] = pd.read_csv("./res/log_spo.csv", header=None)
        # pfyl
        if os.path.isfile("./res/wc_pfyl.csv"):
            dfs["PFYL"] = pd.read_csv("./res/wc_pfyl.csv")
            regret_logs["PFYL"] = pd.read_csv("./res/log_pfyl.csv", header=None)
        # dbb
        if os.path.isfile("./res/wc_dbb.csv"):
            dfs["DBB"] = pd.read_csv("./res/wc_dbb.csv")
            regret_logs["DBB"] = pd.read_csv("./res/log_dbb.csv", header=None)
        # dpo
        if os.path.isfile("./res/wc_dpo.csv"):
            dfs["DPO"] = pd.read_csv("./res/wc_dpo.csv")
            regret_logs["DPO"] = pd.read_csv("./res/log_dpo.csv", header=None)
        # drow learning curve on test set
        wcLearningCurve(regret_logs)
        # draw boxplot of regret per instance
        wcRegret(dfs)
        # draw boxplot of relative regret per instance
        wcRelRegret(dfs)
        # draw boxplot of accuracy per instance
        wcAcc(dfs)
        # draw barplot of optimality ratio per instance
        wcOr(dfs)

# python3 plot.py --plot cmp --prob sp
