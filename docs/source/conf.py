"""Sphinx documentation builder configuration."""

import os
import sys
from importlib.metadata import PackageNotFoundError, version

sys.path.insert(0, os.path.abspath("../../pkg"))

project = "PyTorch-based End-to-End Predict-then-Optimize Tool"
copyright = "2021, Bo Tang"
author = "Bo Tang"

try:
    release = version("pyepo")
except PackageNotFoundError:
    release = "0.0.0+unknown"
version = ".".join(release.split(".")[:2])

# optional solver backends — mock so autoapi doesn't choke when they're absent
autodoc_mock_imports = [
    "coptpy",
    "mpax",
    "jax",
    "jaxlib",
    "gurobipy",
    "pyomo",
    "ortools",
]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "autoapi.extension",
]

# autoapi reads the package AST, so inline type annotations show up automatically
autoapi_dirs = ["../../pkg/pyepo"]
autoapi_python_class_content = "class"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
suppress_warnings = ["autoapi.python_import_resolution"]

# napoleon: nicer rendering for Google/NumPy-style docstrings paired with PEP 484 hints
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True
napoleon_attr_annotations = True

# cross-link external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

highlight_language = "python"
templates_path = ["_templates"]
exclude_patterns = []

html_favicon = "../../images/favicon.ico"
html_theme = "sphinx_rtd_theme"
