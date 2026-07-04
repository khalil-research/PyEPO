Installation
++++++++++++


Pip Install
===========

``PyEPO`` requires Python 3.9 or later. Install from `PyPI <https://pypi.org/project/pyepo>`_ with:

.. code-block:: console

   pip install pyepo



Conda Install
=============

Install from `Anaconda Cloud <https://anaconda.org/pyepo/pyepo>`_ with:

.. code-block:: console

   conda install -c pyepo pyepo


Install from Source
===================

Clone ``PyEPO`` from GitHub:

.. code-block:: console

   git clone -b main --depth 1 https://github.com/khalil-research/PyEPO.git

Install the package from the local checkout:

.. code-block:: console

   pip install ./PyEPO/pkg



Solvers
=======

``PyEPO`` compiles optimization models to a solver backend, so at least one solver must be installed. Each backend has a pip extra that installs its package alongside ``PyEPO``:

* `Gurobi <https://www.gurobi.com/>`_, the default backend. It is commercial, with a free academic license (``pip install pyepo[gurobi]``).
* `COPT <https://www.shanshu.ai/copt>`_, commercial with a free academic license (``pip install pyepo[copt]``).
* `Pyomo <http://www.pyomo.org/>`_, which can use open solvers such as GLPK, CBC, or HiGHS without a commercial license (``pip install pyepo[pyomo]`` plus the solver binary).
* `Google OR-Tools <https://developers.google.com/optimization>`_, open (``pip install pyepo[ortools]``).
* `MPAX <https://github.com/MIT-Lu-Lab/MPAX>`_, open and JAX-based, for GPU and batch solving (``pip install pyepo[mpax]``).

The ``CaVE`` loss also needs Clarabel (``pip install pyepo[cave]``). To install every optional dependency at once, use ``pip install pyepo[all]``.

.. note:: A bare ``pip install pyepo`` does not install a solver backend. The default Gurobi backend needs ``pip install pyepo[gurobi]`` and a Gurobi license. For a license-free setup, use the Pyomo or OR-Tools backend.
