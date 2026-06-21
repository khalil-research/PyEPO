# PyEPO: A PyTorch-based End-to-End Predict-then-Optimize Tool

``PyEPO`` (PyTorch-based End-to-End Predict-then-Optimize Tool) is a Python-based, open-source software that supports modeling and solving predict-then-optimize problems with linear objective functions. The core capability of ``PyEPO`` is to build optimization models with [GurobiPy](https://www.gurobi.com/), [COPT](https://shanshu.ai/copt), [Pyomo](http://www.pyomo.org/), [Google OR-Tools](https://developers.google.com/optimization), [MPAX](https://github.com/MIT-Lu-Lab/MPAX), or any other solvers and algorithms, then embed the optimization model into an artificial neural network for the end-to-end training. For this purpose, ``PyEPO`` implements various methods as [PyTorch](https://pytorch.org/) autograd modules.

## Features

- Implement **SPO+**, **PG**, **DPO** (additive and multiplicative perturbations), **PFYL** (additive and multiplicative perturbations), **I-MLE**, **AI-MLE**, L2-regularized **RFWO/RFYL**, **DBB**, **NID**, **CaVE**, **NCE**, and **LTR**
- Support [Gurobi](https://www.gurobi.com/), [COPT](https://shanshu.ai/copt), [Pyomo](http://www.pyomo.org/), [Google OR-Tools](https://developers.google.com/optimization), and [MPAX](https://github.com/MIT-Lu-Lab/MPAX) API
- Symbolic modeling with `pyepo.dsl`: define an LP, MIP, or QP once, then compile it to any backend
- JAX frontend (`pyepo.func.jax`): train any loss in JAX/Flax with `jax.grad`
- Support parallel computing for optimization solvers
- Support solution caching to speed up training
- Support kNN robust loss to improve decision quality

## CaVE for Binary Linear Programs

For end-to-end learning on **binary linear programs** (TSP, CVRP, knapsack, ...), ``PyEPO`` ships **CaVE**. CaVE replaces the per-step ILP solve with a cone-alignment projection onto the binding-constraint normals at the true optimum, backed by an interior-point QP solver (Clarabel). Because the cone projection is far cheaper than the per-instance ILP solve, CaVE trains an order of magnitude faster than SPO+ at TSP scale.

## GPU-Accelerated Solving with MPAX

``PyEPO`` integrates [MPAX](https://github.com/MIT-Lu-Lab/MPAX), a JAX-based mathematical programming solver using the PDHG algorithm for GPU-accelerated optimization. Key advantages: (1) **GPU-native solving** — the first-order PDHG method runs efficiently on GPU; (2) **batch solving** — an entire mini-batch can be solved simultaneously via vectorization; (3) **no GPU-CPU data transfer overhead** — both the neural network and the solver stay on GPU, eliminating the data transfer bottleneck.

## Documentation

The official docs can be found at [https://khalil-research.github.io/PyEPO](https://khalil-research.github.io/PyEPO).

## Publication

[PyEPO: A PyTorch-based End-to-End Predict-then-Optimize Library for Linear and Integer Programming](https://link.springer.com/article/10.1007/s12532-024-00255-x) (Mathematical Programming Computation)
