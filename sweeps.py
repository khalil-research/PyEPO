#!/usr/bin/env python
# coding: utf-8
"""
PyEPO training with Sweeps
"""

import argparse
import math
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

import pyepo
from config import configs
from run import utils
from run import trainInit

def trainSPO():
    """
    SPO+ train with wandb
    """
    with wandb.init() as run:
        # update config
        config = wandb.config
        wandb.config.update(args)
        print(config)
        # generate data
        data = utils.genData(config)
        if config.prob == "ks":
            config.wght, data = data[0].tolist(), (data[1], -data[2])
        # build model
        model = utils.buildModel(config)
        if config.rel:
            model = model.relax()
        # build data loader
        trainset, testset = utils.buildDataSet(data, model, config)
        trainloader = DataLoader(trainset, batch_size=config.batch, shuffle=True)
        testloader = DataLoader(testset, batch_size=config.batch, shuffle=False)
        # init model
        reg, optimizer = trainInit(config)
        reg.train()
        # device
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        reg.to(device)
        # set SPO+ Loss as criterion
        criterion = pyepo.func.SPOPlus(model, processes=config.proc)
        # train
        time.sleep(1)
        pbar = tqdm(range(config.epoch))
        for e in pbar:
            # load data
            for i, data in enumerate(trainloader):
                x, c, w, z = data
                x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
                # forward pass
                cp = reg(x)
                # spo+
                spoploss = criterion.apply(cp, c, w, z).mean()
                wandb.log({"Train/SPO+": spoploss.item()})
                # l1 reg
                l1_reg = torch.abs(cp - c).sum(dim=1).mean()
                wandb.log({"Train/L1": l1_reg.item()})
                # l2 reg
                l2_reg = ((cp - c) ** 2).sum(dim=1).mean()
                wandb.log({"Train/L2": l2_reg.item()})
                # add hook
                abs_grad = []
                cp.register_hook(lambda grad: abs_grad.append(torch.abs(grad).mean().item()))
                # total loss
                loss = spoploss + config.l1 * l1_reg + config.l2 * l2_reg
                wandb.log({"Train/Total Loss": loss.item()})
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # abs grad
                wandb.log({"Train/Abs Grad": abs_grad[0]})
                # add logs
                desc = "Epoch {}, Loss: {:.4f}".format(e, loss.item())
                pbar.set_description(desc)
            # eval
            if e % 10 == 0:
                regret = pyepo.eval.trueSPO(reg, model, testloader)
                wandb.log({"Regret": regret})
        # eval
        regret = pyepo.eval.trueSPO(reg, model, testloader)
        wandb.log({"Regret": regret})


def trainBB():
    """
    Black-BOX optimizer train with wandb
    """
    with wandb.init() as run:
        # update config
        config = wandb.config
        wandb.config.update(args)
        print(config)
        # generate data
        data = utils.genData(config)
        if config.prob == "ks":
            config.wght, data = data[0].tolist(), (data[1], -data[2])
        # build model
        model = utils.buildModel(config)
        if config.rel:
            model = model.relax()
        # build data loader
        trainset, testset = utils.buildDataSet(data, model, config)
        trainloader = DataLoader(trainset, batch_size=config.batch, shuffle=True)
        testloader = DataLoader(testset, batch_size=config.batch, shuffle=False)
        # init model
        reg, optimizer = trainInit(config)
        reg.train()
        # device
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        reg.to(device)
        # set black-box optimizer
        bb = pyepo.func.blackboxOpt(model, lambd=config.smth, processes=config.proc)
        # set loss
        criterion = torch.nn.L1Loss()
        # train
        time.sleep(1)
        pbar = tqdm(range(config.epoch))
        for e in pbar:
            # load data
            for i, data in enumerate(trainloader):
                x, c, w, z = data
                x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
                # forward pass
                cp = reg(x)
                # black-box optimizer
                wp = bb.apply(cp)
                # objective value
                zp = (wp * c).sum(1).view(-1, 1)
                # SPO loss
                spoloss = criterion(zp, z)
                wandb.log({"Train/SPO": spoloss.item()})
                # l1 reg
                l1_reg = torch.abs(cp - c).sum(dim=1).mean()
                wandb.log({"Train/L1": l1_reg.item()})
                # l2 reg
                l2_reg = ((cp - c) ** 2).sum(dim=1).mean()
                wandb.log({"Train/L2": l2_reg.item()})
                # add hook
                abs_grad = []
                cp.register_hook(lambda grad: abs_grad.append(torch.abs(grad).mean().item()))
                # total loss
                loss = spoloss + config.l1 * l1_reg + config.l2 * l2_reg
                wandb.log({"Train/Total Loss": loss.item()})
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # abs grad
                wandb.log({"Train/Abs Grad": abs_grad[0]})
                # add logs
                desc = "Epoch {}, Loss: {:.4f}".format(e, loss.item())
                pbar.set_description(desc)
            # eval
            if e % 10 == 0:
                regret = pyepo.eval.trueSPO(reg, model, testloader)
                wandb.log({"Regret": regret})
        # eval
        regret = pyepo.eval.trueSPO(reg, model, testloader)
        wandb.log({"Regret": regret})

if __name__ == "__main__":

    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob",
                        type=str,
                        choices=["sp", "ks", "tsp"],
                        help="problem type")
    parser.add_argument("--ksdim",
                        type=int,
                        default=2,
                        help="knapsack dimension")
    parser.add_argument("--mthd",
                        type=str,
                        choices=["spo", "bb"],
                        help="method")
    parser.add_argument("--data",
                        type=int,
                        default=1000,
                        choices=[100, 1000],
                        help="training data size")
    parser.add_argument("--deg",
                        type=int,
                        default=4,
                        help="features polynomial degree")
    parser.add_argument("--noise",
                        type=float,
                        default=0.5,
                        help="noise half-width")
    setting = parser.parse_args()
    # load config
    config = configs[setting.prob][setting.mthd]
    config.proc = 4
    config.data = setting.data # data size
    if config.prob == "ks":
        config.dim = setting.ksdim
    if config.data == 100:
        config.epoch = 300
    if config.data == 1000:
        config.epoch = 100 # epoch
    config.deg = setting.deg # polynomial degree
    config.noise = setting.noise # noise half-width
    # delete parameter to tune
    del config.optm # optimizer
    del config.lr # learning rate
    del config.l1 # l1 regularization
    del config.l2 # l2 regularization
    if setting.mthd == "bb":
        del config.smth # smoothing parameter
    del config.batch # batch size
    # delete unnecessary
    del config.expnum
    del config.timeout
    del config.elog
    del config.path
    # global config
    global args
    args = config

    # init config
    sweep_config = {"name" : "PyEPO-Sweep"}
    # search method
    sweep_config["method"] = "random"
    # metric
    metric = {"name":"Regret", "goal":"minimize"}
    sweep_config["metric"] = metric
    # init parameters
    parameters_dict = {}
    sweep_config["parameters"] = parameters_dict
    parameters_dict["optm"] = {"values":["sgd", "adam"]} # optimizer
    parameters_dict["lr"] = {"distribution":"log_uniform",
                             "min":math.log(1e-4),
                             "max":math.log(1e-1)} # learning rate
    parameters_dict["l1"] = {"distribution":"uniform",
                             "min":0,
                             "max":1e-1} # l1 regularization
    parameters_dict["l2"] = {"distribution":"uniform",
                             "min":0,
                             "max":1e-1} # l2 regularization
    if setting.mthd == "bb":
        parameters_dict["smth"] = {"distribution":"uniform",
                                   "min":10,
                                   "max":20} # smoothing parameter
    parameters_dict["batch"] = {"values":[32, 64, 128]} # batch size

    # init
    sweep_id = wandb.sweep(sweep_config,
                           project="PyEPO-Sweep-{}-{}-d{}p{}e{}".format(config.prob,
                                                                        config.mthd,
                                                                        config.data,
                                                                        config.deg,
                                                                        config.noise))
    # launch agent
    count = 50
    if config.mthd == "spo":
        wandb.agent(sweep_id, function=trainSPO, count=count)
    if config.mthd == "bb":
        wandb.agent(sweep_id, function=trainBB, count=count)
