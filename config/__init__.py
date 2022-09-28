#!/usr/bin/env python
# coding: utf-8
"""
Experiment configuration
"""
from config.config_sp import configSP
from config.config_ks import configKS
from config.config_tsp import configTSP

configs = {}
configs["sp"] = configSP
configs["ks"] = configKS
configs["tsp"] = configTSP
