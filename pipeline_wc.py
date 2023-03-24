#!/usr/bin/env python
# coding: utf-8
"""
Training pipeline for Warcraft
"""

import os
import random
import time

import pyepo
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from run import net
from run import utils
from run import model

# fix random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DATA_DIR = "./data/warcraft_maps/warcraft_shortest_path_oneskin/12x12"
MODEL_DIR = "./model"
if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)
RES_DIR = "./res"
if not os.path.isdir(RES_DIR):
    os.mkdir(RES_DIR)

def pipeline(config):
    # 2s
    if config.mthd == "2s":
        print("Running experiments with two-stage method:")
    # spo
    if config.mthd == "spo":
        print("Running experiments with SPO+ loss:")
    # pfyl
    if config.mthd == "pfyl":
        print("Running experiments with perturbed Fenchel-Youngh loss:")
    # dbb
    if config.mthd == "dbb":
        print("Running experiments for differentiable balck-box optimizer:")
    # travelling salesman
    if config.mthd == "dpo":
        print("Running experiments for differentiable perturbed optimizer:")
    print()
    # load data
    print("Loading data...")
    loader_train, loader_test = getDataLoader(config)
    print()
    # init model
    print("Building convolutional neural network and optimization model...")
    # init net
    nnet = getNet(k=12)
    optmodel = model.shortestPathModel(grid=(12,12))
    print()
    # train
    print("Start Training...")
    regret_log = train(nnet, optmodel, loader_train, loader_test, config)
    # save model
    model_path = MODEL_DIR + "/resnet_{}.pt".format(config.mthd)
    print("Save model to " + model_path + ".")
    torch.save(nnet.state_dict(), model_path)
    # save log
    log_path = RES_DIR + "/log_{}.csv".format(config.mthd)
    print("Save log to " + log_path + ".")
    np.savetxt(log_path, regret_log, delimiter =", ", fmt ='% s')
    print()
    # evaluate
    print("Evaluating...")
    df = evaluate(nnet, optmodel, loader_test)
    res_path = RES_DIR + "/wc_{}.csv".format(config.mthd)
    print("Saved to " + res_path + ".")
    df.to_csv(res_path, index=False)


def getDataLoader(config):
    # maps
    tmaps_train = np.load(DATA_DIR + "/train_maps.npy")
    tmaps_test = np.load(DATA_DIR + "/test_maps.npy")
    # costs
    costs_train = np.load(DATA_DIR + "/train_vertex_weights.npy")
    costs_test = np.load(DATA_DIR + "/test_vertex_weights.npy")
    # paths
    paths_train = np.load(DATA_DIR + "/train_shortest_paths.npy")
    paths_test = np.load(DATA_DIR + "/test_shortest_paths.npy")
    # datasets
    dataset_train = utils.mapDataset(tmaps_train, costs_train, paths_train)
    dataset_test = utils.mapDataset(tmaps_test, costs_test, paths_test)
    # dataloader
    loader_train = DataLoader(dataset_train, batch_size=config.batch, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=config.batch, shuffle=False)
    return loader_train, loader_test


def getNet(k):
    nnet = net.partialResNet(k=k)
    # cuda
    if torch.cuda.is_available():
        nnet = nnet.cuda()
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print("    {}:".format(i), torch.cuda.get_device_name(i))
    else:
        print("Device: CPU")
    return nnet


def train(nnet, optmodel, loader_train, loader_test, config):
    # set optimizer
    optimizer = torch.optim.Adam(nnet.parameters(), lr=config.lr)
    # set loss
    if config.mthd == "2s":
        mseloss = nn.MSELoss()
    if config.mthd == "spo":
        spoploss = pyepo.func.SPOPlus(optmodel, processes=config.proc)
    if config.mthd == "dbb":
        # init dbb
        dbb = pyepo.func.blackboxOpt(optmodel, lambd=config.smth, processes=config.proc)
        # set loss
        class hammingLoss(torch.nn.Module):
            def forward(self, wp, w):
                loss = wp * (1.0 - w) + (1.0 - wp) * w
                return loss.mean(dim=0).sum()
        hmloss = hammingLoss()
    if config.mthd == "dpo":
        # init dpo
        ptb = pyepo.func.perturbedOpt(optmodel, n_samples=config.samp,
                                      sigma=config.sig, processes=config.proc)
        # set loss
        mseloss = nn.MSELoss()
    if config.mthd == "pfyl":
        # set loss
        fyloss = pyepo.func.perturbedFenchelYoung(optmodel, n_samples=config.samp,
                                                  sigma=config.sig, processes=config.proc)
    # init log
    regret_log = [pyepo.metric.regret(nnet, optmodel, loader_test)]
    # train
    tbar = tqdm(range(config.epoch))
    for epoch in tbar:
        nnet.train()
        for x, c, w, z in loader_train:
            # cuda
            if torch.cuda.is_available():
                x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
            # forward pass
            cp = nnet(x) # predicted cost
            if config.mthd == "2s":
                loss = mseloss(cp, c) # loss
            if config.mthd == "spo":
                loss = spoploss(cp, c, w, z).mean() # loss
            if config.mthd == "dbb":
                wp = dbb(cp) # black-box optimizer
                loss = hmloss(wp, w) # loss
            if config.mthd == "dpo":
                we = ptb(cp) # perturbed optimizer
                loss = mseloss(we, w) # loss
            if config.mthd == "pfyl":
                loss = fyloss(cp, w).mean() # loss
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           # log
            tbar.set_description("Epoch: {:2}, Loss: {:3.4f}".format(epoch, loss.item()))
        # scheduled learning rate
        if (epoch == int(config.epoch*0.6)) or (epoch == int(config.epoch*0.8)):
            for g in optimizer.param_groups:
               g['lr'] /= 10
        if epoch % 1 == 0:
            # log regret
            regret = pyepo.metric.regret(nnet, optmodel, loader_test) # regret on test
            regret_log.append(regret)
    return regret_log


def evaluate(nnet, optmodel, dataloader):
    # init data
    data = {"Regret":[], "Relative Regret":[], "Accuracy":[], "Optimal":[]}
    # eval
    nnet.eval()
    for x, c, w, z in tqdm(dataloader):
        # cuda
        if next(nnet.parameters()).is_cuda:
            x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
        # predict
        cp = nnet(x)
        # to numpy
        c = c.to("cpu").detach().numpy()
        w = w.to("cpu").detach().numpy()
        z = z.to("cpu").detach().numpy()
        cp = cp.to("cpu").detach().numpy()
        # solve
        for i in range(cp.shape[0]):
            # sol for pred cost
            optmodel.setObj(cp[i])
            wpi, _ = optmodel.solve()
            # obj with true cost
            zpi = np.dot(wpi, c[i])
            # round
            zpi = zpi.round(1)
            zi = z[i,0].round(1)
            # regret
            regret = (zpi - zi).round(1)
            data["Regret"].append(regret)
            data["Relative Regret"].append(regret / zi)
            # accuracy
            data["Accuracy"].append((abs(wpi - w[i]) < 0.5).mean())
            # optimal
            data["Optimal"].append(abs(regret) < 1e-5)
    # dataframe
    df = pd.DataFrame.from_dict(data)
    # print
    time.sleep(1)
    print("Avg Regret: {:.4f}".format(df["Regret"].mean()))
    print("Avg Rel Regret: {:.2f}%".format(df["Relative Regret"].mean()*100))
    print("Path Accuracy: {:.2f}%".format(df["Accuracy"].mean()*100))
    print("Optimality Ratio: {:.2f}%".format(df["Optimal"].mean()*100))
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # experiments configuration
    parser.add_argument("--mthd",
                        type=str,
                        choices=["2s", "spo", "dbb", "dpo", "pfyl"],
                        help="method")
    # training configuration
    parser.add_argument("--batch",
                        type=int,
                        default=70,
                        help="batch size")
    parser.add_argument("--epoch",
                        type=int,
                        default=50,
                        help="number of epochs")
    parser.add_argument("--optm",
                        type=str,
                        default="adam",
                        choices=["sgd", "adam"],
                        help="optimizer neural network")
    parser.add_argument("--lr",
                        type=float,
                        default=5e-4,
                        help="learning rate")
    parser.add_argument("--smth",
                        type=int,
                        default=10,
                        help="smoothing parameter for Black-Box")
    parser.add_argument("--samp",
                        type=int,
                        default=1,
                        help="number of samples for perturbed methods")
    parser.add_argument("--sig",
                        type=float,
                        default=1.0,
                        help="amplitude parameter for perturbed methods")
    parser.add_argument("--proc",
                        type=int,
                        default=1,
                        help="number of processor for optimization")

    # get configuration
    config = parser.parse_args()

    # run experiment pipeline
    pipeline(config)
