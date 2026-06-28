Installation
++++++++++++


Install from Source
===================

Clone ``PyEPO`` from GitHub.

.. code-block:: console

   git clone -b main --depth 1 https://github.com/khalil-research/PyEPO.git

Install the package from the local checkout.

.. code-block:: console

   pip install PyEPO/pkg/.



Pip Install
===========

Install the `PyPI <https://pypi.org/project/pyepo>`_ release with:

.. code-block:: console

   pip install pyepo



Conda Install
=============

Install the `Anaconda Cloud <https://anaconda.org/pyepo/pyepo>`_ package with:

.. code-block:: console

   conda install -c pyepo pyepo



Solvers
=======

``PyEPO`` compiles optimization models onto a solver backend, so at least one solver must be installed. The default backend is `Gurobi <https://www.gurobi.com/>`_, a commercial solver with a free academic license. The other backends are:

* `COPT <https://www.shanshu.ai/copt>`_, commercial with a free academic license (``pip install coptpy``).
* `Pyomo <http://www.pyomo.org/>`_, which drives open solvers such as GLPK, CBC, or HiGHS with no license (``pip install pyomo`` plus the solver binary).
* `Google OR-Tools <https://developers.google.com/optimization>`_, open (``pip install ortools``).
* `MPAX <https://github.com/MIT-Lu-Lab/MPAX>`_, open and JAX-based, for GPU and batch solving (``pip install mpax``).

.. note:: A bare ``pip install pyepo`` does not install a solver backend. Building a model with the default Gurobi backend then requires a Gurobi license; for a license-free setup, use the Pyomo or OR-Tools backend.
