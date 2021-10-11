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
                    choices=["2s", "spo", "bb"],
                    help="method")
parser.add_argument("--pred2s",
                    type=str,
                    choices=["lr", "rf"],
                    help="predictor for two-stage")
parser.add_argument("--tspform",
                    type=str,
                    choices=["gg", "dfj", "mtz"],
                    help="TSP formulation")
parser.add_argument("--rel",
                    action="store_true",
                    help="train with relaxation model")
setting = parser.parse_args()

# get config
config = configs[setting.prob][setting.mthd]
if setting.prob == "tsp":
    config.form = setting.tspform
if setting.mthd == "2s":
    config.pred = setting.pred2s
config.rel = setting.rel

# test
#def pipeline(alpha):
#    return alpha + 6
#config = 5

# job submission parameters
instance_logs_path = "slurm_logs_spotest"
timeout_min = config.timeout
mem_gb = 4
num_cpus = 32

# config setting
confset = {"data":[100, 1000],
           "noise":[0.0, 0.5],
           "deg":[1, 2, 4, 6]}

jobs = []
for data, noise, deg in itertools.product(*tuple(confset.values())):
    # create executor
    executor = submitit.AutoExecutor(folder=instance_logs_path)
    executor.update_parameters(slurm_additional_parameters={"account": "rrg-khalile2"},
                               timeout_min=timeout_min,
                               mem_gb=mem_gb,
                               cpus_per_task=num_cpus)
    # set config
    config.data = data
    config.noise = noise
    config.deg = deg
    if (setting.mthd != "2s") and (data == 1000):
        if setting.prob == "ks":
            config.epoch = 100
        else:
            config.epoch = 300
    if (setting.mthd != "2s") and (data == 100):
        if setting.prob == "ks":
            config.epoch = 300
        else:
            config.epoch = 1000
    print(config)
    # run job
    job = executor.submit(pipeline, config)
    jobs.append(job)
    print("job_id: {}, mem_gb: {}, num_cpus: {}, logs: {}, timeout: {}" \
          .format(job.job_id, mem_gb, num_cpus, instance_logs_path, timeout_min))

# get outputs
# outputs = [job.result() for job in jobs]
