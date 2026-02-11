#!/usr/bin/env python
# coding: utf-8

import setuptools

long_description = """# PyEPO: A PyTorch-based End-to-End Predict-then-Optimize Tool

``PyEPO`` (PyTorch-based End-to-End Predict-then-Optimize Tool) is a Python-based, open-source software that supports modeling and solving predict-then-optimize problems with linear objective functions. The core capability of ``PyEPO`` is to build optimization models with [GurobiPy](https://www.gurobi.com/), [COPT](https://shanshu.ai/copt), [Pyomo](http://www.pyomo.org/), [Google OR-Tools](https://developers.google.com/optimization), [MPAX](https://github.com/MIT-Lu-Lab/MPAX), or any other solvers and algorithms, then embed the optimization model into an artificial neural network for the end-to-end training. For this purpose, ``PyEPO`` implements various methods as [PyTorch](https://pytorch.org/) autograd modules.

## Features

- Implement **SPO+**, **DBB**, **NID**, **DPO**, **PFYL**, **NCE**, **LTR**, **I-MLE**, **AI-MLE**, and **PG**
- Support [Gurobi](https://www.gurobi.com/), [COPT](https://shanshu.ai/copt), [Pyomo](http://www.pyomo.org/), [Google OR-Tools](https://developers.google.com/optimization), and [MPAX](https://github.com/MIT-Lu-Lab/MPAX) API
- Support parallel computing for optimization solvers
- Support solution caching to speed up training
- Support kNN robust loss to improve decision quality

## GPU-Accelerated Solving with MPAX

``PyEPO`` integrates [MPAX](https://github.com/MIT-Lu-Lab/MPAX), a JAX-based mathematical programming solver using the PDHG algorithm for GPU-accelerated optimization. Key advantages: (1) **GPU-native solving** — the first-order PDHG method runs efficiently on GPU; (2) **batch solving** — an entire mini-batch can be solved simultaneously via vectorization; (3) **no GPU-CPU data transfer overhead** — both the neural network and the solver stay on GPU, eliminating the data transfer bottleneck.

## Documentation

The official docs can be found at [https://khalil-research.github.io/PyEPO](https://khalil-research.github.io/PyEPO).

## Publication

[PyEPO: A PyTorch-based End-to-End Predict-then-Optimize Library for Linear and Integer Programming](https://link.springer.com/article/10.1007/s12532-024-00255-x) (Mathematical Programming Computation)
"""

setuptools.setup(
    # includes all other files
    include_package_data = True,
    # package name
    name = "pyepo",
    # project dir
    packages = setuptools.find_packages(),
    # description
    description = "PyTorch-based End-to-End Predict-then-Optimize Tool",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    license = "MIT",
    keywords = ["predict-then-optimize", "end-to-end", "decision-focused learning",
                "optimization", "deep learning", "pytorch", "linear programming",
                "integer programming"],
    # version
    version = "1.1.1",
    # Github repo
    url = "https://github.com/khalil-research/PyEPO",
    # author name
    author = "Bo Tang",
    # mail address
    author_email = "bolucas.tang@mail.utoronto.ca",
    # restrict Python version
    python_requires = ">=3.7",
    # dependencies
    install_requires = [
        "numpy",
        "scipy",
        "pathos",
        "tqdm",
        "configspace",
        "scikit_learn",
        "torch>=1.13.1"],
    extras_require = {
        "pyomo": ["pyomo>=6.1.2"],
        "gurobi": ["gurobipy>=9.1.2"],
        "ortools": ["ortools>=9.6"],
        "mpax": ["mpax", "jax>=0.4.1", "jaxlib>=0.4.1"],
        "all": ["pyomo>=6.1.2", "gurobipy>=9.1.2", "ortools>=9.6", "mpax", "jax>=0.4.1", "jaxlib>=0.4.1"],
    },
    # classifiers
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research"]
    )
