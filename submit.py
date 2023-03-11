#!/usr/bin/env python
# coding: utf-8
"""
Submit experiments
"""

import argparse
import itertools
import sys
sys.path.append("~/projects/def-khalile2/botang/spo/")

import submitit

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
                    choices=["auto", "lr", "rf", "spo", "dbb", "dpo", "pfyl"],
                    help="method")
parser.add_argument("--ksdim",
                    type=int,
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
parser.add_argument("--sftp",
                    action="store_true",
                    help="positive prediction with SoftPlus activation")
parser.add_argument("--ncpu",
                    type=int,
                    default=32,
                    help="number of CPUs")
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
    config.l1 = 1e-2
if setting.l2:
    config.l2 = 1e-2
config.sftp = setting.sftp

# job submission parameters
instance_logs_path = "slurm_logs_spotest"
# time
timeout_min = config.timeout * config.expnum
# mem & cpu
if setting.mthd in ["auto"]:
    mem_gb = 16
else:
    mem_gb = 8
num_cpus = setting.ncpu

# something to avoid crush
import os
os.environ["OPENBLAS_NUM_THREADS"] = str(num_cpus)

# config setting
confset = {"data":[100, 1000, 5000],
           "noise":[0.0, 0.5],
           "deg":[1, 2, 4, 6]}

jobs = []
for data, noise, deg in itertools.product(*tuple(confset.values())):
    # create executor
    executor = submitit.AutoExecutor(folder=instance_logs_path)
    if data == 5000:
        executor.update_parameters(slurm_additional_parameters={"account": "rrg-khalile2"},
                                   timeout_min=timeout_min,
                                   mem_gb=mem_gb*2,
                                   cpus_per_task=num_cpus)
    else:
        executor.update_parameters(slurm_additional_parameters={"account": "rrg-khalile2"},
                                   timeout_min=timeout_min,
                                   mem_gb=mem_gb,
                                   cpus_per_task=num_cpus)
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
    # test
    #def pipeline(alpha):
    #    return alpha + 6
    #config = 5
    print(config)
    # run job
    job = executor.submit(pipeline, config)
    jobs.append(job)
    print("job_id: {}, mem_gb: {}, num_cpus: {}, logs: {}, timeout: {}" \
          .format(job.job_id, mem_gb, num_cpus, instance_logs_path, timeout_min))

# get outputs
#outputs = [job.result() for job in jobs]
