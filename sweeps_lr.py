#!/usr/bin/env python
# coding: utf-8
"""
PyEPO training with Sweeps
"""

import argparse
import math
import time
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

import pyepo
from config import configs
from run import utils
from run import trainInit

# without this, wandb causes error.
os.environ["WANDB_START_METHOD"] = "thread"

def trainSPO():
    """
    SPO+ train with wandb
    """
    with wandb.init(resume=True) as run:
        # update config
        config = wandb.config
        wandb.config.update(args)
        print(config)
        # generate data
        data = utils.genData(config)
        if config.prob == "ks":
            config.wght, data = data[0].tolist(), (data[1], data[2])
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
        spop = pyepo.func.SPOPlus(model, processes=config.proc)
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
                spoploss = spop(cp, c, w, z).mean()
                wandb.log({"Train/SPO+": spoploss.item()})
                # l1 reg
                l1_reg = torch.abs(cp - c).mean(dim=1).mean()
                wandb.log({"Train/L1": l1_reg.item()})
                # l2 reg
                l2_reg = ((cp - c) ** 2).mean(dim=1).mean()
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
            if (config.data == 1000) and (e % 2 == 0):
                regret = pyepo.metric.regret(reg, model, testloader)
                wandb.log({"Regret": regret})
            if (config.data == 100) and (e % 20 == 0):
                regret = pyepo.metric.regret(reg, model, testloader)
                wandb.log({"Regret": regret})
        # eval
        regret = pyepo.metric.regret(reg, model, testloader)
        wandb.log({"Regret": regret})


def trainDBB():
    """
    Black-BOX optimizer train with wandb
    """
    with wandb.init(resume=True) as run:
        # update config
        config = wandb.config
        wandb.config.update(args)
        print(config)
        # generate data
        data = utils.genData(config)
        if config.prob == "ks":
            config.wght, data = data[0].tolist(), (data[1], data[2])
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
        dbb = pyepo.func.blackboxOpt(model, lambd=config.smth, processes=config.proc)
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
                wp = dbb(cp)
                # objective value
                zp = (wp * c).sum(1).view(-1, 1)
                # loss
                spoloss = criterion(zp, z)
                wandb.log({"Train/SPO": spoloss.item()})
                # l1 reg
                l1_reg = torch.abs(cp - c).mean(dim=1).mean()
                wandb.log({"Train/L1": l1_reg.item()})
                # l2 reg
                l2_reg = ((cp - c) ** 2).mean(dim=1).mean()
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
            if (config.data == 1000) and (e % 2 == 0):
                regret = pyepo.metric.regret(reg, model, testloader)
                wandb.log({"Regret": regret})
            if (config.data == 100) and (e % 20 == 0):
                regret = pyepo.metric.regret(reg, model, testloader)
                wandb.log({"Regret": regret})
        # eval
        regret = pyepo.metric.regret(reg, model, testloader)
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
                        choices=["spo", "dbb"],
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
    if config.prob == "tsp":
        config.form = "dfj"
    if config.data == 100:
        config.epoch = 200
    if config.data == 1000:
        config.epoch = 20 # epoch
    config.deg = setting.deg # polynomial degree
    config.noise = setting.noise # noise half-width
    # delete parameter to tune
    del config.lr # learning rate
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
    sweep_config["method"] = "grid"
    # metric
    metric = {"name":"Regret", "goal":"minimize"}
    sweep_config["metric"] = metric
    # init parameters
    parameters_dict = {}
    sweep_config["parameters"] = parameters_dict
    parameters_dict["lr"] = {"values":[1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]} # learning rate

    # init
    sweep_id = wandb.sweep(sweep_config,
                           project="PyEPO-Sweep_lr-{}-{}-d{}p{}e{}".format(config.prob,
                                                                         config.mthd,
                                                                         config.data,
                                                                         config.deg,
                                                                         config.noise))
    # launch agent
    count = 50
    if config.mthd == "spo":
        wandb.agent(sweep_id, function=trainSPO, count=count)
    if config.mthd == "dbb":
        wandb.agent(sweep_id, function=trainDBB, count=count)
