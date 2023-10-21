#!/usr/bin/env python
# coding: utf-8

import setuptools

setuptools.setup(
    # includes all other files
    include_package_data=True,
    # package name
    name = "pyepo",
    # project dir
    packages = setuptools.find_packages(),
    # description
    description = "PyTorch-based End-to-End Predict-then-Optimize Tool",
    # version
    version = "0.3.5",
    # Github repo
    url = "https://github.com/khalil-research/PyEPO",
    # author name
    author = "Bo Tang",
    # mail address
    author_email = "bolucas.tang@mail.utoronto.ca",
    # dependencies
    install_requires = [
        "numpy",
        "scipy",
        "pathos",
        "tqdm",
        "Pyomo>=6.1.2",
        "gurobipy>=9.1.2",
        "scikit_learn",
        "torch>=1.13.1"],
   # classifiers
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"]
    )
