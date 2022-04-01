# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.append(os.path.abspath("../../pkg"))
#sys.path.append(os.path.abspath("../../pyepo"))
sys.path.append(os.path.abspath("../../pkg/pyepo/data"))
sys.path.append(os.path.abspath("../../pkg/pyepo/model"))
sys.path.append(os.path.abspath("../../pkg/pyepo/twostage"))
sys.path.append(os.path.abspath("../../pkg/pyepo/func"))
sys.path.append(os.path.abspath("../../pkg/pyepo/train"))
sys.path.append(os.path.abspath("../../pkg/pyepo/eval"))


# -- Project information -----------------------------------------------------

project = "PyTorch-based End-to-End Predict-then-Optimize Tool"
copyright = "2021, Bo Tang"
author = "Bo Tang"

# The full version, including alpha/beta/rc tags
release = "v0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = ["sphinx.ext.autodoc",
              "autoapi.extension",
              "sphinx.ext.autosummary",
              "sphinx.ext.doctest",
              "sphinx.ext.intersphinx",
              "sphinx.ext.todo",
              "sphinx.ext.coverage",
              "sphinx.ext.mathjax",
              "sphinx.ext.napoleon"
              ]

# To document __init__()
autoclass_content = "both"

# Add any path that contain packages here, relative to this directory.
autoapi_dirs = ["../../pkg/pyepo"]

# Turn on autosummary
autosummary_generate = True

# Add defualt programming language
hightlight_language = "python"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_build", "_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# icon
html_favicon = "../../images/favicon.ico"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
