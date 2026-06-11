#!/usr/bin/env python
"""Benchmark cut recycling: repeated TSP-DFJ solves with and without the lazy cut pool."""

import time

import numpy as np

from pyepo.model.grb.tsp import tspDFJModel


def bench(recycle: bool, num_nodes: int = 20, num_solves: int = 50, seed: int = 42):
    rng = np.random.RandomState(seed)
    model = tspDFJModel(num_nodes=num_nodes, recycle_cuts=recycle)
    costs = rng.rand(num_solves, model.num_cost)
    tick = time.perf_counter()
    objs = []
    for c in costs:
        model.setObj(c)
        _, obj = model.solve()
        objs.append(obj)
    return time.perf_counter() - tick, objs


if __name__ == "__main__":
    t_off, objs_off = bench(recycle=False)
    t_on, objs_on = bench(recycle=True)
    # recycling must not change any optimum
    assert np.allclose(objs_off, objs_on, atol=1e-6), "objective mismatch"
    print(f"recycle_cuts=False: {t_off:.2f}s")
    print(f"recycle_cuts=True:  {t_on:.2f}s  ({t_off / t_on:.2f}x speedup)")
