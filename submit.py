#!/usr/bin/env python
# coding: utf-8
"""
Submit experiments
"""
import submitit

def add(a, b):
    return a + b
args_dict = {"a":1, "b":9}

# job submission parameters
instance_logs_path = "slurm_logs_spotest"
slurm_output_dir = "slurm_spotest"
timeout_min = 1
mem_gb = 8
num_cpus = 32

executor = submitit.AutoExecutor(folder=instance_logs_path)
executor.update_parameters(slurm_additional_parameters={"account": "def-khalile2"},
                           timeout_min=timeout_min,
                           mem_gb=mem_gb,
                           cpus_per_task=num_cpus)
job = executor.submit(add, args_dict)
print("job_id: {}, mem_gb: {}, num_cpus: {}, logs: {}" \
      .format(job.job_id, mem_gb, num_cpus, instance_logs_path))
