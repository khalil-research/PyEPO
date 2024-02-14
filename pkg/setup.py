#!/usr/bin/env python
# coding: utf-8

import setuptools

long_description = "``PyEPO`` (PyTorch-based End-to-End Predict-then-Optimize Tool) is a Python-based, open-source software that supports modeling and solving predict-then-optimize problems with the linear objective function. The core capability of ``PyEPO`` is to build optimization models with [GurobiPy](https://www.gurobi.com/), [Pyomo](http://www.pyomo.org/), or any other solvers and algorithms, then embed the optimization model into an artificial neural network for the end-to-end training. For this purpose, ``PyEPO`` implements various methods as [PyTorch](https://pytorch.org/) autograd modules."

setuptools.setup(
    # includes all other files
    include_package_data=True,
    # package name
    name = "pyepo",
    # project dir
    packages = setuptools.find_packages(),
    # description
    description = "PyTorch-based End-to-End Predict-then-Optimize Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # version
    version = "0.3.6",
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
