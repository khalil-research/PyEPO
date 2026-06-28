"""Sphinx documentation builder configuration."""

import os
import sys
from importlib.metadata import PackageNotFoundError, version

sys.path.insert(0, os.path.abspath("../../pkg"))

project = "PyEPO: A PyTorch/JAX-based End-to-End Predict-then-Optimize Tool"
copyright = "2021, Bo Tang"
author = "Bo Tang"

try:
    release = version("pyepo")
except PackageNotFoundError:
    release = "0.0.0+unknown"
version = ".".join(release.split(".")[:2])

# optional solver backends; mock so autoapi doesn't choke when they're absent
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

# autoapi reads the package AST, so inline type annotations appear in generated docs
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
html_theme = "furo"
html_theme_options = {
    # palette derived from the PyEPO logo:
    #   carrot-orange as the primary brand color (links, headings accents)
    #   leaf-green appears on `.. tip::` admonitions and on autoapi signature names
    "light_css_variables": {
        "color-brand-primary": "#B95B10",
        "color-brand-content": "#B95B10",
        "color-admonition-title--tip": "#1F8237",
        "color-admonition-title-background--tip": "#E8F5EA",
        "color-api-name": "#1F8237",
    },
    "dark_css_variables": {
        "color-brand-primary": "#F4B452",
        "color-brand-content": "#F4B452",
        "color-admonition-title--tip": "#7CD688",
        "color-admonition-title-background--tip": "rgba(124, 214, 136, 0.12)",
        "color-api-name": "#7CD688",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
