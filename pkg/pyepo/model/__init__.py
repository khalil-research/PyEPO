#!/usr/bin/env python
# coding: utf-8
"""
Optimization Model based on solvers
"""

from pyepo.model import opt
try:
    from pyepo.model import grb
except:
    pass
try:
    from pyepo.model import copt
except:
    pass
try:
    from pyepo.model import omo
except:
    pass
