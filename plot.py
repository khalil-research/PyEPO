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
from matplotlib import patheffects as path_effects
from matplotlib import ticker
from matplotlib.colors import colorConverter
import tol_colors as tc

from run import utils
from config import configs


def getConfig(prob):
    config = {}
    config["lr"]   = configs[prob]["lr"]
    config["rf"]   = configs[prob]["rf"]
    config["auto"] = configs[prob]["auto"]
    config["spo"]  = configs[prob]["spo"]
    config["pfyl"] = configs[prob]["pfyl"]
    config["dbb"]  = configs[prob]["dbb"]
    config["dpo"]  = configs[prob]["dpo"]
    return config


def getDf(config, degs, mthd, col="True SPO"):
    dfs = pd.DataFrame()
    for deg in degs:
        config[mthd].deg = deg
        path = utils.getSavePath(config[mthd])
        df = pd.read_csv(path)
        dfs[deg] = df[col]
    return dfs

def getRowSP(grids, n, d, e):
    regrets = {"lr":{"mean":[],"std":[]}, "rf":{"mean":[],"std":[]}, "spo":{"mean":[],"std":[]}, "dbb":{"mean":[],"std":[]}}
    elapses = {"lr":{"mean":[],"std":[]}, "rf":{"mean":[],"std":[]}, "spo":{"mean":[],"std":[]}, "dbb":{"mean":[],"std":[]}}
    for g in grids:
        l, t = getRowDataSP(g, n, d, e)
        for m in regrets:
            regrets[m]["mean"].append(l[m]["mean"])
            regrets[m]["std"].append(l[m]["std"])
            elapses[m]["mean"].append(t[m]["mean"])
            elapses[m]["std"].append(t[m]["std"])
    return regrets, elapses


def getRowDataSP(grid, n, d, e):
    # dir
    dir_name = "./res/sp/h{}w{}/gurobi".format(grid[0], grid[1])
    file_name = {}
    file_name["lr"] = "n{}p5-d{}-e{}_2s-lr.csv".format(n,d,e)
    file_name["rf"] = "n{}p5-d{}-e{}_2s-rf.csv".format(n,d,e)
    file_name["spo"] = "n{}p5-d{}-e{}_spo_lr_adam0.01_bs32_l10.0l20.0_c8.csv".format(n,d,e)
    file_name["dbb"] = "n{}p5-d{}-e{}_dbb_lr_adam0.1_bs32_l10.0l20.0_c8-lamb20.csv".format(n,d,e)
    # load data
    regret, elapse = {}, {}
    for m, f in file_name.items():
        df = pd.read_csv(dir_name + "/" + f)
        regret[m] = {"mean":df["Unamb SPO"].mean(), "std":df["Unamb SPO"].std()}
        elapse[m] = {"mean":df["Elapsed"].mean(), "std":df["Elapsed"].std()}
    return regret, elapse

def getRowKS(dims, n, d, e):
    regrets = {"lr":{"mean":[],"std":[]}, "rf":{"mean":[],"std":[]},
               "spo":{"mean":[],"std":[]}, "dbb":{"mean":[],"std":[]}
              }
    for dim in dims:
        l = getRowDataKS(dim, n, d, e)
        for m in regrets:
            regrets[m]["mean"].append(l[m]["mean"])
            regrets[m]["std"].append(l[m]["std"])
    return regrets


def getRowDataKS(dim, n, d, e):
    # dir
    dir_name = "./res/ks/i32d{}c20/gurobi".format(dim)
    file_name = {}
    file_name["lr"] = "n{}p5-d{}-e{}_2s-lr.csv".format(n,d,e)
    file_name["rf"] = "n{}p5-d{}-e{}_2s-rf.csv".format(n,d,e)
    file_name["spo"] = "n{}p5-d{}-e{}_spo_lr_adam0.01_bs32_l10.0l20.0_c32.csv".format(n,d,e)
    file_name["dbb"] = "n{}p5-d{}-e{}_dbb_lr_adam0.1_bs32_l10.0l20.0_c32-lamb10.csv".format(n,d,e)
    # load data
    regret = {}
    for m, f in file_name.items():
        df = pd.read_csv(dir_name + "/" + f)
        regret[m] = {"mean":df["Unamb SPO"].mean(), "std":df["Unamb SPO"].std()}
    return regret


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
    return colorsys.hls_to_rgb(c[0],1-amount * (1-c[1]),c[2])


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


def regPlot(config, data, noise):
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
    colors = [cset.mint, cset.pink, cset.orange, cset.light_blue]
    # get df
    df_spo  = getDf(config, degs, "spo")
    df_spo1 = getDf(config, degs, "spo l1")
    df_spo2 = getDf(config, degs, "spo l2")
    df_dbb  = getDf(config, degs, "dbb")
    df_dbb1 = getDf(config, degs, "dbb l1")
    df_dbb2 = getDf(config, degs, "dbb l2")
    # draw boxplot
    fig = plt.figure(figsize=(16,6))
    c = colors[2]
    bp1 = plt.boxplot(df_spo, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_spo.shape[1])-0.4, widths=0.12)
    c = colors[2]
    bp2 = plt.boxplot(df_spo1, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_spo.shape[1])-0.24, widths=0.12)
    for box in bp2['boxes']:
        box.set(hatch="++++", fill=False)
    c = colors[2]
    bp3 = plt.boxplot(df_spo2, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_spo.shape[1])-0.08, widths=0.12)
    for box in bp3["boxes"]:
        box.set(hatch="OO", fill=False)
    c = colors[3]
    bp4 = plt.boxplot(df_dbb, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_dbb.shape[1])+0.08, widths=0.12)
    c = colors[3]
    bp5 = plt.boxplot(df_dbb1, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_dbb.shape[1])+0.24, widths=0.12)
    for box in bp5["boxes"]:
        box.set(hatch="++++", fill=False)
    c = colors[3]
    bp6 = plt.boxplot(df_dbb2, boxprops=dict(facecolor=c, color=c, linewidth=4), medianprops=dict(color="w", linewidth=2),
                      whiskerprops=dict(color=c, linewidth=2), capprops=dict(color=c, linewidth=2),
                      flierprops=dict(markeredgecolor=c, marker="o", markersize=5, markeredgewidth=2),
                      patch_artist=True, positions=np.arange(df_dbb.shape[1])+0.4, widths=0.12)
    for box in bp6["boxes"]:
        box.set(hatch="OO", fill=False)
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
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    plt.title("Shortest Path\nTraining Set Size = {}, Noise Half−width = {}".format(data, noise), fontsize=30)
    plt.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0], bp4["boxes"][0], bp5["boxes"][0], bp6["boxes"][0]],
               ["SPO+","SPO+ L1","SPO+ L2","DBB","DBB L1","DBB L2"], fontsize=22, loc=2, labelspacing=0.2,
               handlelength=1, ncol=2)
    # save
    dir = "./images/reg-{}-n{}e{}.png".format(prob, data, int(10*noise))
    fig.savefig(dir, dpi=300)
    print("Saved to " + dir)


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
    cmap = tc.tol_cmap("sunset")(np.linspace(0, 1, 11))
    colors = {"lr":cset.mint,
              "rf":cset.pink,
              "auto":cset.pear,
              "spo":cset.orange,
              "spo l1":cset.orange,
              "spo l2":cset.orange,
              "spo rel":cmap[6],
              "spo rel(gg)":cmap[7],
              "spo rel(mtz)":cmap[6],
              "dbb":cset.light_blue,
              "dbb l1":cset.light_blue,
              "dbb l2":cset.light_blue,
              "dbb rel":cmap[4],
              "dbb rel(gg)":cmap[3],
              "dbb rel(mtz)":cmap[4],}
    w = colorConverter.to_rgba("w", alpha=0.6) # white
    # get df
    dfs = {}
    for mthd in config:
        path = utils.getSavePath(config[mthd])
        dfs[mthd] = pd.read_csv(path)
    # draw boxplot
    fig, ax = plt.subplots(figsize=(12,12))
    # init xmax & ymax
    xmax, ymax = 0, 0
    for mthd in dfs:
        df, c = dfs[mthd], colorConverter.to_rgba(colors[mthd], alpha=0.6)
        x, y = df["MSE"].mean(), df["True SPO"].mean()
        xmax, ymax = max(x, xmax), max(y, ymax)
        size = int((np.log(df["Elapsed"].mean())+4)*500)
        if mthd.split(" ")[-1] == "l1":
            ax.scatter(x, y, s=size, color=c, marker="o", hatch="++++", facecolor=w)
        elif mthd.split(" ")[-1] == "l2":
            ax.scatter(x, y, s=size, color=c, marker="o", hatch="OO", facecolor=w)
        else:
            ax.scatter(x, y, s=size, color=c, marker="o")
        # annotate
        txt = ax.annotate(mthd+" :{:.2f} Sec".format(df["Elapsed"].mean()), (x,y), fontsize=20, color=colors[mthd])
        txt.set_path_effects([path_effects.withStroke(linewidth=0.75, foreground='k')])
    plt.xlabel("Mean Squared Error", fontsize=36)
    plt.xticks(fontsize=24)
    plt.ylabel("Normalized Regret", fontsize=36)
    plt.yticks(fontsize=24)
    plt.xlim(0.0, xmax*1.1)
    plt.ylim(0.0, ymax*1.1)
    plt.title("Training Set Size = {}, Polynomial degree = {}, Noise Half−width = {}" \
              .format(data, deg, noise), fontsize=24)
    # save
    dir = "./images/td-{}-n{}e{}.png".format(prob, data, int(10*noise))
    fig.savefig(dir, dpi=300)
    print("Saved to " + dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--plot",
                        type=str,
                        choices=["cmp", "rel", "reg", "trd", "scl"],
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
        # add reg
        config["spo l1"] = deepcopy(config["spo"])
        config["spo l1"].l1 = 1e-3
        config["spo l2"] = deepcopy(config["spo"])
        config["spo l2"].l2 = 1e-3
        config["dbb l1"] = deepcopy(config["dbb"])
        config["dbb l1"].l1 = 1e-3
        config["dbb l2"] = deepcopy(config["dbb"])
        config["dbb l2"].l2 = 1e-3
        # delete 2s
        del config["lr"]
        del config["rf"]
        del config["auto"]
        # varying setting
        confset = {"data":[100, 1000],
                   "noise":[0.0, 0.5]}
        # plot
        for data, noise in itertools.product(*tuple(confset.values())):
            regPlot(config, data, noise)

    ############################################################################
    # regularization
    if setting.plot == "trd":
        # get config
        config = getConfig(setting.prob)
        # add reg
        config["spo l1"] = deepcopy(config["spo"])
        config["spo l1"].l1 = 1e-3
        config["spo l2"] = deepcopy(config["spo"])
        config["spo l2"].l2 = 1e-3
        config["dbb l1"] = deepcopy(config["dbb"])
        config["dbb l1"].l1 = 1e-3
        config["dbb l2"] = deepcopy(config["dbb"])
        config["dbb l2"].l2 = 1e-3
        # add relaxation
        if setting.prob == "ks":
            config["spo rel"] = deepcopy(config["spo"])
            config["spo rel"].rel = True
            config["dbb rel"] = deepcopy(config["dbb"])
            config["dbb rel"].rel = True
        if setting.prob == "tsp":
            # add relaxation
            config["spo rel(mtz)"] = deepcopy(config["spo"])
            config["spo rel(mtz)"].form = "mtz"
            config["spo rel(mtz)"].rel = True
            config["dbb rel(mtz)"] = deepcopy(config["dbb"])
            config["dbb rel(mtz)"].form = "mtz"
            config["dbb rel(mtz)"].rel = True
            config["spo rel(gg)"] = deepcopy(config["spo"])
            config["spo rel(gg)"].form = "gg"
            config["spo rel(gg)"].rel = True
            config["dbb rel(gg)"] = deepcopy(config["dbb"])
            config["dbb rel(gg)"].form = "gg"
            config["dbb rel(gg)"].rel = True
            config["spo"].form = "dfj"
            config["dbb"].form = "dfj"
        # varying setting
        confset = {"data":[100, 1000],
                   "noise":[0.5]}
        # plot
        for data, noise in itertools.product(*tuple(confset.values())):
            tradeoffPlot(config, data, noise)

# python3 plot.py --plot cmp --prob sp
